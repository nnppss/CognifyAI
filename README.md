# CognifyAI

CognifyAI turns a lecture video into a study workspace. It combines transcript text, on-screen text pulled from video frames, and AI-generated study tools so you can ask questions, review faster, and revisit key concepts.

## What It Does

- Process a YouTube lecture or a local video upload
- Build a searchable lecture context from transcript + OCR text
- Let you ask questions about the lecture
- Generate summaries, detailed notes, flashcards, and quizzes
- Download summaries and notes as PDF files
- Save processed lectures in a local library dashboard for later review

## Installation

You need `ffmpeg`, `tesseract`, and Python 3 installed before running the app.

### macOS

1. Install system dependencies:

   ```bash
   brew install ffmpeg tesseract
   ```

2. If `python3` is not already installed:

   ```bash
   brew install python
   ```

3. Create a virtual environment and install Python packages:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Windows

1. Install Python 3 if it is not already installed. Make sure Python is added to `PATH`.

2. Install system dependencies:

   ```powershell
   winget install -e --id Gyan.FFmpeg
   winget install -e --id UB-Mannheim.TesseractOCR
   ```

3. Open a new terminal so the installed tools are available on `PATH`.

4. Create a virtual environment and install Python packages:

   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

   If you use Command Prompt instead of PowerShell:

   ```bat
   .\.venv\Scripts\activate.bat
   pip install -r requirements.txt
   ```

### Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
SPEECHMATICS_API_KEY=your_speechmatics_api_key
GEMINI_MODEL_NAME=gemini-2.5-flash
```

- `GEMINI_API_KEY` is used for answers, summaries, notes, quizzes, and flashcards
- `SPEECHMATICS_API_KEY` is used for uploaded videos and YouTube fallback transcription
- `GEMINI_MODEL_NAME` is optional

## Running

### macOS

```bash
source .venv/bin/activate
python3 run.py
```

### Windows

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python run.py
```

Command Prompt:

```bat
.\.venv\Scripts\activate.bat
python run.py
```

Open `http://127.0.0.1:8000` in your browser.

## How To Use

1. Open the app in your browser.
2. Paste a YouTube lecture URL or upload a local video file.
3. Wait for the lecture to finish processing.
4. Use the generated lecture workspace to:
   - ask questions in Q&A
   - generate a summary
   - generate detailed notes
   - create flashcards
   - generate a quiz
5. Reopen saved lectures from the Library Dashboard.

## Notes

- Local uploads support `mp4`, `mkv`, `avi`, and `mov`
- Maximum upload size is `500 MB`
- Generated files and study data are stored locally in `data/`
- If YouTube download fails with a format/runtime error, install `Node.js` or `Deno` and try again
