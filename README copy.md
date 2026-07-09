# JARVIS Voice Assistant

A Python voice assistant using speech recognition, text-to-speech, and the OpenAI API.

## Setup

1. **Clone the repo**
   ```
   git clone https://github.com/ImRidwane/JARVIS.git
   cd JARVIS
   ```

2. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```
   > Note: `pyaudio` can be tricky to install on some systems. On Windows, try `pipwin install pyaudio` if the normal install fails.

3. **Add your API keys**

   Copy the example env file:
   ```
   cp .env.example .env
   ```

   Open `.env` and fill in your own keys:
   ```
   OPENAI_API_KEY=your-openai-api-key-here
   WEATHER_API_KEY=your-weatherapi-key-here
   WEATHER_LOCATION=New York
   ```

   - Get an OpenAI key at https://platform.openai.com/api-keys
   - Get a free WeatherAPI key at https://www.weatherapi.com/

   **`.env` is gitignored and will never be committed.** Never share your real keys or commit them to the repo.

4. **Run JARVIS**
   ```
   python jarvis.py
   ```

## Voice commands

- "What's the time?"
- "What's the weather?"
- "List files" / "Create file" / "Delete file"
- "Goodbye" to exit
- Anything else gets routed to the AI assistant for a conversational response

## Security note

This project uses environment variables for all API keys. If you fork this repo, never hardcode your keys directly into `jarvis.py` — always use the `.env` file so your credentials stay local and out of git history.
