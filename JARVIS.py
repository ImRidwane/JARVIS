import speech_recognition as sr
import pyttsx3
import requests
import openai
from datetime import datetime
import sys

# Initialize the speech recognition and text-to-speech engines
r = sr.Recognizer()
engine = pyttsx3.init()

# Set up your OpenAI API key
# Replace with your own OpenAI API key
openai.api_key = "your-openai-api-key-here"

def speak(text):
    """Convert text to speech."""
    engine.say(text)
    engine.runAndWait()

def listen():
    """Listen for user input and recognize speech."""
    with sr.Microphone() as source:
        print("Listening...")
        try:
            input_audio = r.listen(source)
            text = r.recognize_google(input_audio)
            print("Command: " + text)
            return text.lower()
        except sr.UnknownValueError:
            print("Sorry, I didn't catch that. Could you please repeat?")
            return ""
        except sr.RequestError as e:
            print("Uh oh! An error occurred: " + str(e))
            return ""

def chat_with_user(user_input):
    """Use ChatGPT API to generate a conversational response."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are JARVIS, a helpful assistant."},
                {"role": "user", "content": user_input},
            ],
            temperature=0.7,
            max_tokens=150,
        )
        return response["choices"][0]["message"]["content"]
    except openai.error.AuthenticationError:
        return "Authentication failed. Please check your API key."
    except openai.error.RateLimitError:
        return "Rate limit exceeded. Please try again later."
    except openai.error.OpenAIError as e:
        return f"OpenAI error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"

def get_current_time():
    """Fetch the current time."""
    now = datetime.now()
    return now.strftime("%I:%M %p")  # Format: Hour:Minute AM/PM

def get_weather(location="New York"):
    """Fetch current weather using a weather API."""
    # Replace with your own API key and URL
    api_key = "your-weather-api-key-here"
    base_url = "https://api.weatherapi.com/v1/current.json"

    params = {
        "q": location,
        "key": api_key,
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise HTTPError for bad responses
        weather_data = response.json()
        
        # Extract relevant information
        temp = weather_data["current"]["temp_f"]  # Adjust based on your API response structure
        description = weather_data["current"]["condition"]["text"]
        
        return f"The current temperature at your location is {temp}°F with {description}."
    except requests.exceptions.RequestException as e:
        return f"Sorry, I couldn't fetch the weather. Error: {e}"
    
def process_command(command):
    """Process the recognized command and perform actions."""
    print(f"Processing command: {command}")  # Debugging line
    if "what" in command or "how" in command or "tell me" in command:
        speak("Let me check that for you.")
        response = chat_with_user(command)
        speak(response)
    
    elif "hello" in command:
        speak("Hello! How can I assist you today?")

    elif "time" in command or "current time" in command:
        current_time = datetime.now().strftime("%I:%M %p")
        speak(f"The current time is {current_time}")

    elif "weather" in command or "current temperature" in command:
        location = "New York"  # You can change or dynamically extract location
        weather_report = get_weather(location)
        speak(weather_report)

    elif "goodbye" in command:
        speak("Goodbye! Have a great day!")
        sys.exit()

    else:
        speak("I'm sorry, I didn't understand that command.")

# Main loop
while True:
    command = listen()
    if command:  # Process only if command is not empty
        process_command(command)
