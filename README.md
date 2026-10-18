# Lucy Voice Companion — Local Starter

Lucy was created by **Abhimanyu Rana**. It is a single-user local FastAPI app with a social-style chat screen, an animated call page, browser speech input, and optional AI, voice, memory, and web-search features. API keys stay on the Python backend in `.env`; do not place them in `index.html` or commit the real `.env` file.

## Quick start

1. Install Python 3.10 or newer.
2. Open a terminal in the project folder and create a virtual environment:

   ```bash
   python -m venv .venv
   # Windows PowerShell: .venv\Scripts\Activate.ps1
   # Windows cmd.exe: .venv\Scripts\activate.bat
   # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env`.
4. Add at least `GROQ_API_KEY` to enable generated replies. Add the other keys for their optional features.
5. Start the local server:

   ```bash
   python main.py
   ```

6. Open `http://127.0.0.1:7860/` on the same computer running the server. Accept microphone permission if using voice typing or calls.

### Using Lucy on a phone

Lucy is a browser-based web app, not a native Android/iPhone app. For phone use, deploy the Python backend and web page to a host with an **HTTPS** URL, then open that URL in the phone's browser and allow microphone access. The local address `127.0.0.1` refers only to the device running the server, and microphone access generally requires a secure context such as HTTPS.

## Configuration (`.env`)

| Variable | Purpose |
| --- | --- |
| `GROQ_API_KEY` | Required for AI-generated replies. Without it Lucy uses basic demo replies. |
| `GROQ_MODEL` | Optional chat model override. |
| `HF_TOKEN` | Optional. Extracts tone and simple non-sensitive facts after each reply. |
| `HF_MODEL` | Optional memory model override. |
| `SERPER_API_KEY` | Optional. Enables **Search web** in the **+** menu. |
| `SERPER_GL`, `SERPER_HL` | Search country (`in`) and language (`en`). |
| `SARVAM_API_KEY` | Optional. Enables Lucy's generated voice and the call speech-recognition fallback. |
| `SARVAM_TTS_MODEL`, `SARVAM_TTS_SPEAKER` | Optional voice overrides. |
| `PORT` | Defaults to `7860`. |
| `DB_PATH` | Defaults to `lucy.db` in the project folder. |

A missing optional key disables just that feature. The browser's built-in voice fallback varies by browser and device.

## What the features do

- **Memory:** when `HF_TOKEN` is set, each new user message is analyzed after the reply for explicitly stated, non-sensitive facts (a preferred name, hobby, interest). Facts are saved in the local SQLite `memories` table and may be used in later replies. `GET /api/memories` lists them and `DELETE /api/memories` clears them.
- **Web search:** with `SERPER_API_KEY` set and **Search web** turned on, the top results are given to Lucy as reference context and shown as clickable sources under her answer. Off by default.
- **Voice:** with `SARVAM_API_KEY` set, Lucy's replies are spoken through `/api/tts`. Special symbols, emoji, and URLs are removed from spoken audio while remaining visible in chat. If generated voice is slow, browser speech starts instead.
- **Call input:** prefers browser speech recognition. If unavailable and `SARVAM_API_KEY` is set, the call records one short turn, detects the pause, and transcribes it through `/api/stt`. This fallback requires HTTPS and microphone permission. The call is turn-based, not full-duplex.
- **Delete chat** (in the **+** menu) clears the visible conversation and recent context but leaves saved memories alone.

## Speed

- **Streaming replies:** the first sentence is sent to the voice as soon as it is written, while the rest is still being generated. Each following sentence is voiced in parallel, so there are no gaps.
- **Faster listening:** the browser microphone sends your words about 0.65 s after you stop changing them instead of waiting for the browser's slow end-of-speech signal. The recorder fallback stops after 0.7 s of silence, and the mic restarts within 0.1-0.15 s.
- **Shorter call context:** calls send only the last 4 turns, which shortens the wait for the first word.
- If your voice gets cut off mid-sentence, raise `END_OF_SPEECH_MS` in `index.html` (for example to 900).

## Test each feature

- **Backend and key status:** open `http://127.0.0.1:7860/api/health`. It shows which keys are present and how many memories are saved; it makes no billable request.
- **AI replies:** send Lucy a normal message. If the key is missing, you get demo mode.
- **Memory:** say "My name is Rahul and I like cricket." then open `/api/memories`.
- **Search:** tap **+**, enable **Search web**, ask a current factual question.
- **Voice:** enable sound and receive a reply, or test directly:

  ```bash
  curl -X POST http://127.0.0.1:7860/api/tts \
    -H "Content-Type: application/json" \
    -d '{"text":"Namaste, main Lucy hoon.","language_code":"hi-IN"}' \
    --output lucy-test.mp3
  ```

- **Clear saved memories:** `curl -X DELETE http://127.0.0.1:7860/api/memories`
- **Identity check:** ask Lucy "Tumhe kisne banaya?" — she answers that Abhimanyu Rana made her.

## Safety, privacy, and deployment

The optional **Mature chat · Adults 18+** control is off by default and asks for confirmation before it turns on; it is not age verification. Minors, age-ambiguous people, coercion, force, exploitation, abuse, incapacitation, and illegal sexual content remain out of bounds, and safety rules always apply.

The local server binds to `127.0.0.1`. It has no user authentication, multi-user isolation, rate limiting, or production abuse monitoring. Before public deployment, add those controls, use HTTPS, verify age where legally required, and keep the database private.

## API routes

- `GET /` — chat screen.
- `GET /call` — call screen route.
- `GET /api/health` — configuration status.
- `POST /api/chat` — AI reply in one piece, optional memory analysis, optional web search.
- `POST /api/chat/stream` — same reply streamed sentence by sentence; the app uses this so Lucy's voice starts on the first sentence.
- `POST /api/tts` — speech generation.
- `POST /api/stt` — speech transcription fallback for call audio.
- `GET /api/memories` and `DELETE /api/memories` — list or clear saved memories.

## Main files

- `main.py` — FastAPI routes and integrations.
- `index.html` — chat theme, composer tools, speech controls, and animated call page.
- `lucy-avatar.png` — Lucy's profile photo.
- `.env.example` — copy to `.env` and fill in keys.
- `requirements.txt` — Python dependencies.