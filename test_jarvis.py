"""Offline checks for jarvis.py - no real API calls, no microphone."""
import json
import os
import sys
import tempfile
from types import SimpleNamespace

STATE = tempfile.mkdtemp()
os.environ.update(
    OPENAI_API_KEY="test-key",
    TOMORROW_API_KEY="test-weather-key",
    JARVIS_STATE_DIR=STATE,
    JARVIS_VOICE="0",
    WEATHER_LOCATION="Harrisburg, PA",
)

import jarvis  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"  -> {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


# --- stub the weather HTTP layer -------------------------------------------
REALTIME = {
    "data": {
        "time": "2026-11-18T21:00:00Z",
        "values": {
            "temperature": 47.3,
            "temperatureApparent": 41.0,
            "humidity": 62,
            "windSpeed": 8.4,
            "weatherCode": 1000,
        },
    }
}
FORECAST = {
    "timelines": {
        "daily": [
            {"time": "2026-11-19T05:00:00Z", "values": {"temperatureMax": 52, "temperatureMin": 38, "weatherCodeMax": 4001, "precipitationProbabilityAvg": 70}},
            {"time": "2026-11-20T05:00:00Z", "values": {"temperatureMax": 49, "temperatureMin": 35, "weatherCodeMax": 1101, "precipitationProbabilityAvg": 20}},
        ]
    }
}


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


calls_made = []


def fake_get(url, params=None, timeout=None):
    calls_made.append((url, dict(params or {})))
    if "realtime" in url:
        return FakeResponse(REALTIME)
    return FakeResponse(FORECAST)


jarvis.requests.get = fake_get

# --- tool-level checks -----------------------------------------------------
weather = jarvis.tool_get_weather()
check("weather uses tomorrow.io realtime endpoint",
      calls_made and "tomorrow.io/v4/weather/realtime" in calls_made[0][0], calls_made)
check("weather sends imperial units", calls_made[0][1].get("units") == "imperial", calls_made[0][1])
check("weather sends api key", calls_made[0][1].get("apikey") == "test-weather-key")
check("weather reports temp", "47F" in weather, weather)
check("weather decodes code 1000 to clear", "clear" in weather, weather)
check("weather includes humidity", "62%" in weather, weather)

forecast = jarvis.tool_get_forecast(days=2)
check("forecast hits forecast endpoint", any("forecast" in c[0] for c in calls_made))
check("forecast mentions rain", "rain" in forecast, forecast)
check("forecast lists two days", forecast.count("2026-11") == 2, forecast)

# --- sandbox checks --------------------------------------------------------
out = jarvis.dispatch_tool("read_file", {"path": "../../../../etc/passwd"})
check("path traversal blocked", out.startswith("Error"), out)

out = jarvis.dispatch_tool("run_command", {"name": "rm -rf /"})
check("unapproved command refused", "not an approved command" in out, out)

out = jarvis.dispatch_tool("create_file", {"path": "notes/hello.txt", "content": "hi"})
check("create_file writes inside workspace", os.path.exists(os.path.join(jarvis.WORKSPACE, "notes/hello.txt")), out)

out = jarvis.dispatch_tool("get_weather", {"bogus_arg": 1})
check("bad tool args returned as text, not raised", out.startswith("Error"), out)

# --- memory checks ---------------------------------------------------------
jarvis.tool_remember("Lives in Harrisburg, PA")
jarvis.tool_remember("Prefers short answers")
check("memory persisted to disk", len(json.load(open(jarvis.MEMORY_FILE))) == 2)
prompt = jarvis.build_system_prompt()
check("memory injected into system prompt", "Harrisburg" in prompt and "short answers" in prompt)
jarvis.tool_forget("short answers")
check("forget removes one fact", len(json.load(open(jarvis.MEMORY_FILE))) == 1)

# --- agent loop: tool call then answer ------------------------------------
script = [
    SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(name="get_weather", arguments='{"location": "Harrisburg, PA"}'),
            )
        ],
    ),
    SimpleNamespace(content="It's 47F and clear in Harrisburg, PA.", tool_calls=None),
]
seen = []


class FakeCompletions:
    def create(self, **kwargs):
        seen.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=script[len(seen) - 1])])


jarvis.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

reply, history = jarvis.run_agent_turn("what's the weather like?", [])
check("agent returns final text", reply == "It's 47F and clear in Harrisburg, PA.", reply)
check("agent made two model calls", len(seen) == 2, len(seen))
check("tools were offered to the model", seen[0]["tools"] is jarvis.TOOLS)
check("tool result fed back to model",
      any(m.get("role") == "tool" and "47F" in str(m.get("content")) for m in seen[1]["messages"]),
      [m.get("role") for m in seen[1]["messages"]])
check("history has assistant+tool+assistant",
      [m["role"] for m in history] == ["user", "assistant", "tool", "assistant"],
      [m["role"] for m in history])
check("transcript logged", os.path.exists(jarvis.TRANSCRIPT_FILE))

# --- step limit ------------------------------------------------------------
looping = SimpleNamespace(
    content=None,
    tool_calls=[SimpleNamespace(id="c", function=SimpleNamespace(name="get_current_time", arguments="{}"))],
)


class LoopCompletions:
    def create(self, **kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=looping)])


jarvis.client = SimpleNamespace(chat=SimpleNamespace(completions=LoopCompletions()))
reply, _ = jarvis.run_agent_turn("loop forever", [], max_steps=3)
check("step limit stops a runaway loop", "step limit" in reply, reply)

# --- history trimming keeps tool pairs intact -----------------------------
jarvis.MAX_HISTORY_MESSAGES = 4
long_hist = [
    {"role": "user", "content": "a"},
    {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
    {"role": "tool", "tool_call_id": "1", "content": "x"},
    {"role": "assistant", "content": "b"},
    {"role": "user", "content": "c"},
    {"role": "assistant", "content": "d"},
]
trimmed = jarvis.trim_history(long_hist)
check("trim never starts with an orphaned tool result", trimmed[0]["role"] != "tool", [m["role"] for m in trimmed])

print()
print("FAILURES:", FAILS if FAILS else "none")
sys.exit(1 if FAILS else 0)
