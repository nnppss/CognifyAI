<div align="center">

# CognifyAI

Turn lecture videos into a searchable AI study workspace.

`Flask` `Gemini` `RAG` `OCR` `YouTube` `Local Video`

</div>

## What It Is

CognifyAI processes a lecture video and turns it into a study workspace. It combines transcript text, on-screen text extracted from video frames, and AI-generated study tools so you can ask questions, review faster, and revisit key concepts later.

## Highlights

- Process a YouTube lecture or a local video upload
- Combine spoken transcript text and visual OCR text into one searchable lecture context
- Ask lecture-specific questions with timestamp-aware answers
- Generate summaries, detailed notes, flashcards, and quizzes
- Download summaries and notes as PDF files
- Save processed lectures locally in a library dashboard for later review

## Tech Stack

| Layer | Tools | Purpose |
| --- | --- | --- |
| Backend | Flask, Jinja2 | Web app routes, templates, and server logic |
| LLM | Google Gemini | Answers, summaries, notes, quizzes, and flashcards |
| Retrieval | Sentence Transformers, Rank-BM25, NumPy | Embeddings, hybrid search, and ranking |
| Transcript + Media | yt-dlp, youtube-transcript-api, Speechmatics, FFmpeg | Video input, transcript fetching, fallback transcription, audio/frame extraction |
| OCR | Tesseract, pytesseract, Pillow | Extract on-screen text from sampled lecture frames |
| Output + Storage | ReportLab, local JSON/NPY/PDF files in `data/` | PDF export and local persistence |

## Workflow

1. Add a lecture using a YouTube URL or a local video upload.
2. Fetch the transcript from YouTube, or fall back to Speechmatics transcription when needed.
3. Sample video frames with FFmpeg and extract on-screen text with Tesseract OCR.
4. Merge transcript text and OCR text into lecture chunks, then build searchable indexes and embeddings.
5. Retrieve the most relevant chunks with hybrid search and use Gemini to answer questions.
6. Generate summaries, notes, flashcards, quizzes, and PDFs from the processed lecture.
7. Save everything locally so the lecture can be reopened from the library dashboard.

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
6. Download summary or notes as PDF when needed.

## Notes

- Local uploads support `mp4`, `mkv`, `avi`, and `mov`
- Maximum upload size is `500 MB`
- Generated files and study data are stored locally in `data/`
- No separate database setup is required

## Troubleshooting

- If `ffmpeg` or `tesseract` is not found, reinstall it and open a new terminal window
- If `winget` is unavailable on Windows, install FFmpeg and Tesseract manually and add them to `PATH`
- If YouTube download fails with a format or runtime error, install `Node.js` or `Deno` and try again
- If the app says an API key is missing, check the `.env` file and restart the server
