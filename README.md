# JARVIS

A voice-controlled personal assistant, rebuilt as a tool-using agent. Started as a
high-school project; v3 replaces the keyword router with a real agent loop.

Ask it something and the model decides which tools to call, reads the results, and
answers. Adding a capability means adding one tool, not another `elif`.

## How it works

```
you -> agent loop -> model -> tool calls -> tool results -> model -> answer
```

The loop is `run_agent_turn()` in `jarvis.py`. It is deliberately small:

1. Send the system prompt (including remembered facts) plus recent history.
2. If the model asks for tools, run them and feed the results back.
3. Repeat until it answers or hits `JARVIS_MAX_STEPS`.
4. Trim history so the context window stays bounded.

## Tools

| Tool | What it does |
| --- | --- |
| `get_weather` | Current conditions from Tomorrow.io |
| `get_forecast` | Daily forecast, up to 7 days |
| `get_current_time` | Local date and time |
| `list_files` / `read_file` | Inspect the workspace |
| `create_file` | Write a text file |
| `delete_file` | Delete a file (asks first) |
| `run_command` | One of a fixed allow-list of read-only commands (asks first) |
| `remember` / `forget` | Durable facts about you |

## Safety

- **Sandboxed paths.** File tools resolve every path and refuse anything outside
  `JARVIS_WORKSPACE`. `../../etc/passwd` is rejected.
- **No shell interpolation.** Commands are argv vectors run with `shell=False` from a
  fixed allow-list. The model can pick a name; it cannot supply the command.
- **Approval gate.** Destructive tools (`delete_file`, `run_command`) ask out loud and
  need a yes.
- **Errors go back to the model.** A failed tool returns text, so the agent can recover
  instead of crashing.
- **Secrets stay in `.env`.** Nothing is hard-coded; `.env` is gitignored.

## Setup

```bash
git clone https://github.com/ImRidwane/JARVIS.git
cd JARVIS
pip install -r requirements.txt
cp .env.example .env      # then add your keys
```

You need an [OpenAI key](https://platform.openai.com/api-keys) and a
[Tomorrow.io key](https://app.tomorrow.io/development/keys).

## Run

```bash
python jarvis.py                       # voice loop, falls back to typing
python jarvis.py --text                # typed input, no microphone
python jarvis.py --once "weather in Harrisburg?"
```

Voice is optional. `SpeechRecognition` and `pyttsx3` are imported lazily, so the agent
runs headless on a machine with no audio stack. If the microphone is missing, it asks
you to type instead of dying.

## State

Both live outside the process, so nothing is lost on restart:

- `~/.jarvis/memory.json` - facts the model chose to remember, injected into every prompt
- `~/.jarvis/transcript.jsonl` - append-only log of turns, tool calls and results

## Requirements

- Python 3.9+
- `openai`, `requests`, `python-dotenv`
- Optional: `SpeechRecognition`, `pyttsx3`, `pyaudio` for voice

## License

MIT
