# CognifyAI: AI-Powered Learning Companion

This repository contains the source code for **CognifyAI**, a project designed to transform lecture videos into interactive learning companions.

## Project Overview

CognifyAI processes a given lecture video by extracting both spoken captions and on-screen text from key frames. It organizes this information into a searchable knowledge base. When a user asks a question, the system retrieves the most relevant context from the video and uses a Large Language Model (LLM) to generate an intelligent, context-aware answer.

This code provides a structural overview and simulation of the project's workflow, intended for demonstration purposes.

## How to Run

1.  Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
2.  Set required API keys:
    ```bash
    export GEMINI_API_KEY="your_gemini_api_key"
    export SPEECHMATICS_API_KEY="your_speechmatics_api_key"
    ```
    Or create a local `.env` file in the project root:
    ```env
    GEMINI_API_KEY=your_gemini_api_key
    SPEECHMATICS_API_KEY=your_speechmatics_api_key
    ```
    Optional: override the default Gemini model:
    ```env
    GEMINI_MODEL_NAME=gemini-2.5-flash
    ```
3.  Run the web app:
    ```bash
    python3 server.py
    ```
4.  Open `http://127.0.0.1:8000` in your browser.
