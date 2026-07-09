"""
JARVIS Voice Assistant - v2
Updated to use the modern OpenAI SDK (>=1.0.0), environment-based
API key management, conversation memory, streaming responses, and
a safer approach to system commands.
"""

import os
import sys
import subprocess
from datetime import datetime

import requests
import speech_recognition as sr
import pyttsx3
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()  # reads variables from a local .env file (never commit this file)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
WEATHER_LOCATION = os.getenv("WEATHER_LOCATION", "New York")

if not OPENAI_API_KEY:
    sys.exit("Missing OPENAI_API_KEY. Add it to a .env file before running JARVIS.")

client = OpenAI(api_key=OPENAI_API_KEY)

r = sr.Recognizer()
engine = pyttsx3.init()

# Rolling conversation history so JARVIS can handle follow-up questions
conversation_history = [
    {"role": "system", "content": "You are JARVIS, a concise, helpful voice assistant."}
]
MAX_HISTORY_MESSAGES = 12  # keep the context window small and fast

# Only these commands can be executed via voice - never shell out to
# arbitrary user-provided text.
SAFE_SYSTEM_COMMANDS = {
    "list directory": ["dir"] if sys.platform == "win32" else ["ls"],
    "current directory": ["cd"] if sys.platform == "win32" else ["pwd"],
}


# ---------------------------------------------------------------------------
# Core I/O
# ---------------------------------------------------------------------------

def speak(text: str) -> None:
    """Convert text to speech."""
    print(f"JARVIS: {text}")
    engine.say(text)
    engine.runAndWait()


def listen() -> str:
    """Listen for user input and recognize speech."""
    with sr.Microphone() as source:
        print("Listening...")
        r.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = r.listen(source)
            text = r.recognize_google(audio)
            print(f"You: {text}")
            return text.lower()
        except sr.UnknownValueError:
            print("Sorry, I didn't catch that. Could you please repeat?")
            return ""
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            return ""


# ---------------------------------------------------------------------------
# Chat (modern OpenAI SDK, streaming, with memory)
# ---------------------------------------------------------------------------

def chat_with_user(user_input: str) -> str:
    """Use the OpenAI API to generate a conversational, context-aware response."""
    conversation_history.append({"role": "user", "content": user_input})

    # trim history so it doesn't grow unbounded (keep system msg + recent turns)
    trimmed = [conversation_history[0]] + conversation_history[-MAX_HISTORY_MESSAGES:]

    try:
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=trimmed,
            temperature=0.7,
            max_tokens=150,
            stream=True,
        )

        full_reply = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full_reply += delta

        conversation_history.append({"role": "assistant", "content": full_reply})
        return full_reply

    except Exception as e:
        return f"Sorry, I hit an error talking to OpenAI: {e}"


# ---------------------------------------------------------------------------
# Utility commands
# ---------------------------------------------------------------------------

def get_current_time() -> str:
    return datetime.now().strftime("%I:%M %p")


def get_weather(location: str = None) -> str:
    """Fetch current weather using WeatherAPI."""
    if not WEATHER_API_KEY:
        return "Weather isn't configured. Add WEATHER_API_KEY to your .env file."

    location = location or WEATHER_LOCATION
    base_url = "https://api.weatherapi.com/v1/current.json"
    params = {"q": location, "key": WEATHER_API_KEY}

    try:
        response = requests.get(base_url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        temp = data["current"]["temp_f"]
        description = data["current"]["condition"]["text"]
        return f"It's currently {temp}°F with {description} in {location}."
    except requests.exceptions.RequestException as e:
        return f"Sorry, I couldn't fetch the weather. Error: {e}"


def handle_file_management(command: str) -> None:
    """Handle simple, contained file operations in the working directory only."""
    if "create file" in command:
        filename = "new_file.txt"
        with open(filename, "w") as f:
            f.write("This is a new file.")
        speak(f"A file named {filename} has been created.")

    elif "delete file" in command:
        filename = "new_file.txt"
        if os.path.exists(filename):
            os.remove(filename)
            speak(f"The file {filename} has been deleted.")
        else:
            speak(f"The file {filename} does not exist.")

    elif "list files" in command:
        files = os.listdir(".")
        speak(f"The files in this directory are: {', '.join(files)}")


def handle_system_commands(command: str) -> None:
    """Execute only pre-approved, allow-listed system commands - never raw shell input."""
    for phrase, cmd_list in SAFE_SYSTEM_COMMANDS.items():
        if phrase in command:
            try:
                output = subprocess.check_output(cmd_list, text=True, shell=False)
                speak(f"Here's the result: {output}")
            except subprocess.CalledProcessError as e:
                speak(f"An error occurred while running the command: {e}")
            return
    speak("That system command isn't in my approved list.")


# ---------------------------------------------------------------------------
# Command routing
# ---------------------------------------------------------------------------

def process_command(command: str) -> None:
    print(f"Processing command: {command}")

    if "goodbye" in command or "shut down" in command:
        speak("Goodbye! Have a great day!")
        sys.exit()

    elif "hello" in command or "hey jarvis" in command:
        speak("Hello! How can I assist you today?")

    elif "time" in command:
        speak(f"The current time is {get_current_time()}")

    elif "weather" in command or "temperature" in command:
        speak(get_weather())

    elif "run command" in command:
        handle_system_commands(command)

    elif "file" in command:
        handle_file_management(command)

    elif command.strip():
        # default: let the model handle open-ended questions
        speak(chat_with_user(command))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    speak("JARVIS is online.")
    while True:
        command = listen()
        if command:
            process_command(command)
