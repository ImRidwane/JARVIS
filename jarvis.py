"""
JARVIS - a tool-using voice agent.

v3 changes, versus the v2 keyword router:

  * The model decides what to do. Tools are declared once (TOOLS) and the
    agent loop lets the model call them, read the results and iterate until
    it can answer. Adding a capability means adding one tool, not another
    elif branch.
  * Weather now comes from Tomorrow.io (realtime + daily forecast).
  * Durable memory: facts the model chooses to keep are stored in
    ~/.jarvis/memory.json and injected into the system prompt on every run.
  * State lives outside the process: memory and a JSONL transcript survive
    restarts instead of dying with the conversation list.
  * Destructive tools are gated behind an explicit yes, and shell access is
    an allow-list of argv vectors - never a raw string handed to a shell.
  * File tools are sandboxed to JARVIS_WORKSPACE.
  * Voice is optional. SpeechRecognition / pyttsx3 are imported lazily, so the
    agent runs headless on a machine with no microphone or audio stack.

Run:
    python jarvis.py                 # voice loop (falls back to text)
    python jarvis.py --text          # force typed input
    python jarvis.py --once "weather in Harrisburg?"
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TOMORROW_API_KEY = os.getenv("TOMORROW_API_KEY") or os.getenv("WEATHER_API_KEY")
WEATHER_LOCATION = os.getenv("WEATHER_LOCATION", "Harrisburg, PA")

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_STEPS = int(os.getenv("JARVIS_MAX_STEPS", "6"))
MAX_HISTORY_MESSAGES = int(os.getenv("JARVIS_MAX_HISTORY", "16"))

STATE_DIR = Path(os.getenv("JARVIS_STATE_DIR", str(Path.home() / ".jarvis")))
MEMORY_FILE = STATE_DIR / "memory.json"
TRANSCRIPT_FILE = STATE_DIR / "transcript.jsonl"

WORKSPACE = Path(os.getenv("JARVIS_WORKSPACE", os.getcwd())).resolve()

VOICE_ENABLED = os.getenv("JARVIS_VOICE", "1") != "0"

if not OPENAI_API_KEY:
    sys.exit("Missing OPENAI_API_KEY. Add it to a .env file before running JARVIS.")

client = OpenAI(api_key=OPENAI_API_KEY)

BASE_SYSTEM_PROMPT = (
    "You are JARVIS, a concise, dependable personal assistant. "
    "You can call tools to check the weather, read and write files, run a small "
    "set of approved commands, and remember durable facts about the user. "
    "Prefer calling a tool over guessing: if the user asks about the weather or "
    "about a file, look it up rather than assuming. "
    "When the user tells you something worth keeping long term (their city, their "
    "name, a preference, a recurring commitment), call remember. "
    "Keep spoken answers short - one to three sentences unless asked for detail. "
    "Never claim you did something unless a tool result confirms it."
)


# ---------------------------------------------------------------------------
# State: memory + transcript
# ---------------------------------------------------------------------------

def _ensure_state_dir() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)


def load_memory() -> list[str]:
    try:
        return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []


def save_memory(facts: list[str]) -> None:
    _ensure_state_dir()
    MEMORY_FILE.write_text(json.dumps(facts, indent=2), encoding="utf-8")


def build_system_prompt() -> str:
    facts = load_memory()
    if not facts:
        return BASE_SYSTEM_PROMPT
    rendered = "\n".join(f"- {fact}" for fact in facts)
    return f"{BASE_SYSTEM_PROMPT}\n\nThings you know about the user:\n{rendered}"


def log_turn(role: str, content: str, extra: dict[str, Any] | None = None) -> None:
    """Append-only JSONL transcript, so history survives a restart."""
    _ensure_state_dir()
    record = {"ts": datetime.now().isoformat(timespec="seconds"), "role": role, "content": content}
    if extra:
        record.update(extra)
    with TRANSCRIPT_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# I/O: speech in, speech out (both optional)
# ---------------------------------------------------------------------------

_recognizer = None
_tts = None


def speak(text: str) -> None:
    print(f"JARVIS: {text}")
    global _tts
    if not VOICE_ENABLED:
        return
    try:
        if _tts is None:
            import pyttsx3  # imported lazily so headless runs still work

            _tts = pyttsx3.init()
        _tts.say(text)
        _tts.runAndWait()
    except Exception:
        # No audio stack (container, CI, missing drivers) - printing is enough.
        pass


def listen() -> str:
    """Return the user's next utterance, or "" if nothing was heard."""
    global _recognizer
    if not VOICE_ENABLED:
        return input("You: ").strip()
    try:
        if _recognizer is None:
            import speech_recognition as sr

            _recognizer = sr.Recognizer()
        import speech_recognition as sr

        with sr.Microphone() as source:
            print("Listening...")
            _recognizer.adjust_for_ambient_noise(source, duration=0.4)
            audio = _recognizer.listen(source, timeout=8, phrase_time_limit=15)
        text = _recognizer.recognize_google(audio)
        print(f"You: {text}")
        return text.strip()
    except Exception:
        # Mic missing, silence, or recognition failure: fall back to typing.
        return input("You (typed): ").strip()


def confirm(question: str) -> bool:
    """Explicit approval gate for anything destructive."""
    speak(f"{question} Say yes to confirm.")
    answer = listen().strip().lower()
    return answer in {"yes", "y", "yeah", "confirm", "do it", "sure"}


# ---------------------------------------------------------------------------
# Sandboxed path handling
# ---------------------------------------------------------------------------

def _safe_path(raw: str) -> Path:
    """Resolve a path and refuse anything outside JARVIS_WORKSPACE."""
    candidate = (WORKSPACE / raw).resolve() if not os.path.isabs(raw) else Path(raw).resolve()
    if candidate != WORKSPACE and WORKSPACE not in candidate.parents:
        raise ValueError(f"Path '{raw}' is outside the workspace ({WORKSPACE}).")
    return candidate


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

WEATHER_CODES = {
    0: "unknown conditions",
    1000: "clear",
    1001: "cloudy",
    1100: "mostly clear",
    1101: "partly cloudy",
    1102: "mostly cloudy",
    2000: "fog",
    2100: "light fog",
    4000: "drizzle",
    4001: "rain",
    4200: "light rain",
    4201: "heavy rain",
    5000: "snow",
    5001: "flurries",
    5100: "light snow",
    5101: "heavy snow",
    6000: "freezing drizzle",
    6001: "freezing rain",
    6200: "light freezing rain",
    6201: "heavy freezing rain",
    7000: "ice pellets",
    7101: "heavy ice pellets",
    7102: "light ice pellets",
    8000: "thunderstorms",
}


def _tomorrow_get(path: str, location: str, extra: dict[str, str] | None = None) -> dict:
    if not TOMORROW_API_KEY:
        raise RuntimeError(
            "Weather isn't configured. Add TOMORROW_API_KEY to your .env file."
        )
    params = {
        "location": location,
        "apikey": TOMORROW_API_KEY,
        "units": "imperial",
    }
    if extra:
        params.update(extra)
    response = requests.get(
        f"https://api.tomorrow.io/v4/weather/{path}",
        params=params,
        timeout=10,
    )
    if response.status_code == 401:
        raise RuntimeError("Tomorrow.io rejected the API key (401). Check TOMORROW_API_KEY.")
    if response.status_code == 429:
        raise RuntimeError("Tomorrow.io rate limit hit (429). Try again shortly.")
    response.raise_for_status()
    return response.json()


def tool_get_weather(location: str | None = None) -> str:
    """Current conditions from Tomorrow.io."""
    place = location or WEATHER_LOCATION
    payload = _tomorrow_get("realtime", place)
    values = payload.get("data", {}).get("values", {})
    temp = values.get("temperature")
    feels = values.get("temperatureApparent")
    humidity = values.get("humidity")
    wind = values.get("windSpeed")
    condition = WEATHER_CODES.get(int(values.get("weatherCode", 0)), "unknown conditions")

    parts = [f"It's {temp:.0f}F and {condition} in {place}"]
    if feels is not None and abs(feels - (temp or 0)) >= 3:
        parts.append(f"feels like {feels:.0f}F")
    if humidity is not None:
        parts.append(f"humidity {humidity:.0f}%")
    if wind is not None:
        parts.append(f"wind {wind:.0f} mph")
    return ", ".join(parts) + "."


def tool_get_forecast(location: str | None = None, days: int = 3) -> str:
    """Daily forecast from Tomorrow.io."""
    place = location or WEATHER_LOCATION
    payload = _tomorrow_get("forecast", place, {"timelines": "daily"})
    daily = payload.get("timelines", {}).get("daily", [])
    if not daily:
        return f"No forecast data came back for {place}."

    lines = []
    for entry in daily[: max(1, min(days, 7))]:
        values = entry.get("values", {})
        when = entry.get("time", "")[:10]
        high = values.get("temperatureMax")
        low = values.get("temperatureMin")
        condition = WEATHER_CODES.get(int(values.get("weatherCodeMax", 0)), "unknown")
        rain = values.get("precipitationProbabilityAvg")
        line = f"{when}: {condition}, high {high:.0f}F / low {low:.0f}F"
        if rain is not None:
            line += f", {rain:.0f}% chance of precipitation"
        lines.append(line)
    return f"Forecast for {place}:\n" + "\n".join(lines)


def tool_get_current_time() -> str:
    return datetime.now().strftime("%A, %B %d %Y at %I:%M %p")


def tool_list_files(path: str = ".") -> str:
    target = _safe_path(path)
    if not target.is_dir():
        return f"{target} is not a directory."
    entries = sorted(target.iterdir())[:200]
    if not entries:
        return f"{target} is empty."
    return "\n".join(f"{e.name}/" if e.is_dir() else e.name for e in entries)


def tool_read_file(path: str, max_chars: int = 4000) -> str:
    target = _safe_path(path)
    if not target.is_file():
        return f"No file at {target}."
    text = target.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        return text[:max_chars] + f"\n... [truncated, {len(text)} chars total]"
    return text


def tool_create_file(path: str, content: str = "") -> str:
    target = _safe_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} characters to {target}."


def tool_delete_file(path: str) -> str:
    target = _safe_path(path)
    if not target.exists():
        return f"No file at {target}."
    if not confirm(f"Delete {target}?"):
        return "Cancelled by the user."
    target.unlink()
    return f"Deleted {target}."


# Allow-listed shell commands. Keys are what the model may ask for; values are
# argv vectors run with shell=False. Nothing here interpolates user text.
SAFE_COMMANDS: dict[str, list[str]] = {
    "list_directory": ["ls", "-la"],
    "disk_usage": ["df", "-h"],
    "uptime": ["uptime"],
    "python_version": [sys.executable, "--version"],
    "git_status": ["git", "status", "--short"],
}


def tool_run_command(name: str) -> str:
    argv = SAFE_COMMANDS.get(name)
    if not argv:
        return (
            f"'{name}' is not an approved command. "
            f"Approved: {', '.join(sorted(SAFE_COMMANDS))}."
        )
    if not confirm(f"Run '{name}'?"):
        return "Cancelled by the user."
    try:
        output = subprocess.run(
            argv, capture_output=True, text=True, timeout=20, cwd=str(WORKSPACE), shell=False
        )
        return (output.stdout or output.stderr or "(no output)").strip()[:2000]
    except subprocess.TimeoutExpired:
        return f"'{name}' timed out after 20 seconds."


def tool_remember(fact: str) -> str:
    facts = load_memory()
    if fact in facts:
        return "Already remembered."
    facts.append(fact.strip())
    save_memory(facts[-50:])
    return f"Remembered: {fact.strip()}"


def tool_forget(fact: str) -> str:
    facts = load_memory()
    needle = fact.strip().lower()
    remaining = [f for f in facts if needle not in f.lower()]
    if len(remaining) == len(facts):
        return "Nothing matched that."
    save_memory(remaining)
    return f"Forgot {len(facts) - len(remaining)} matching fact(s)."


# ---------------------------------------------------------------------------
# Tool schemas (what the model sees)
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather conditions for a location, via Tomorrow.io.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and state/country, e.g. 'Harrisburg, PA'. Omit to use the configured default.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_forecast",
            "description": "Multi-day daily weather forecast for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and state/country."},
                    "days": {"type": "integer", "description": "Number of days, 1-7. Default 3."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Current local date and time on this machine.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Directory, default '.'"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create or overwrite a text file inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Delete a file inside the workspace. Requires user confirmation.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run one of a fixed set of read-only commands. "
                f"Approved names: {', '.join(sorted(SAFE_COMMANDS))}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": sorted(SAFE_COMMANDS)},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Save a durable fact about the user that should persist across sessions.",
            "parameters": {
                "type": "object",
                "properties": {"fact": {"type": "string"}},
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget",
            "description": "Remove a previously remembered fact.",
            "parameters": {
                "type": "object",
                "properties": {"fact": {"type": "string"}},
                "required": ["fact"],
            },
        },
    },
]

TOOL_IMPLS: dict[str, Callable[..., str]] = {
    "get_weather": tool_get_weather,
    "get_forecast": tool_get_forecast,
    "get_current_time": tool_get_current_time,
    "list_files": tool_list_files,
    "read_file": tool_read_file,
    "create_file": tool_create_file,
    "delete_file": tool_delete_file,
    "run_command": tool_run_command,
    "remember": tool_remember,
    "forget": tool_forget,
}


def dispatch_tool(name: str, arguments: dict[str, Any]) -> str:
    """Run one tool call. Never raises: errors go back to the model as text."""
    impl = TOOL_IMPLS.get(name)
    if impl is None:
        return f"Error: no tool named '{name}'."
    try:
        return str(impl(**arguments))
    except TypeError as exc:
        return f"Error: bad arguments for {name}: {exc}"
    except Exception as exc:
        return f"Error running {name}: {exc}"


# ---------------------------------------------------------------------------
# The agent loop
# ---------------------------------------------------------------------------

def run_agent_turn(
    user_input: str,
    history: list[dict[str, Any]],
    max_steps: int = MAX_STEPS,
) -> tuple[str, list[dict[str, Any]]]:
    """Let the model think, call tools, read results, and answer.

    Returns (final_reply, updated_history). This is the whole difference
    between a chat script and an agent: a bounded loop around tool calls.
    """
    history.append({"role": "user", "content": user_input})
    log_turn("user", user_input)

    for step in range(max_steps):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": build_system_prompt()}] + history,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.4,
        )
        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None)

        if not tool_calls:
            reply = (message.content or "").strip() or "I don't have an answer for that."
            history.append({"role": "assistant", "content": reply})
            log_turn("assistant", reply, {"steps": step + 1})
            return reply, history

        history.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in tool_calls
                ],
            }
        )

        for call in tool_calls:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            result = dispatch_tool(call.function.name, arguments)
            log_turn("tool", result, {"tool": call.function.name, "arguments": arguments})
            history.append(
                {"role": "tool", "tool_call_id": call.id, "content": result}
            )

    exhausted = (
        "I hit my step limit without finishing that. Try breaking it into a smaller ask."
    )
    history.append({"role": "assistant", "content": exhausted})
    log_turn("assistant", exhausted, {"steps": max_steps, "stopped": "step_limit"})
    return exhausted, history


def trim_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep recent turns, and never orphan a tool result from its call."""
    if len(history) <= MAX_HISTORY_MESSAGES:
        return history
    trimmed = history[-MAX_HISTORY_MESSAGES:]
    while trimmed and trimmed[0].get("role") == "tool":
        trimmed.pop(0)
    return trimmed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = sys.argv[1:]

    if "--once" in args:
        idx = args.index("--once")
        prompt = " ".join(args[idx + 1 :]).strip()
        if not prompt:
            sys.exit("Usage: python jarvis.py --once \"your question\"")
        reply, _ = run_agent_turn(prompt, [])
        speak(reply)
        return

    if "--text" in args:
        global VOICE_ENABLED
        VOICE_ENABLED = False

    speak("JARVIS is online.")
    history: list[dict[str, Any]] = []

    while True:
        command = listen()
        if not command:
            continue
        if command.lower().strip(" .!") in {"goodbye", "shut down", "exit", "quit", "bye"}:
            speak("Goodbye.")
            return
        try:
            reply, history = run_agent_turn(command, history)
            history = trim_history(history)
            speak(reply)
        except KeyboardInterrupt:
            speak("Goodbye.")
            return
        except Exception as exc:
            speak(f"Something went wrong: {exc}")


if __name__ == "__main__":
    main()
