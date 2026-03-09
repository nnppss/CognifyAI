import os


def _load_dotenv_if_exists() -> None:
    """
    Load key=value pairs from a local .env file into os.environ if present.
    Existing environment variables are not overwritten.
    """
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return

    try:
        with open(env_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key:
                    os.environ.setdefault(key, value)
    except Exception:
        # If .env parsing fails, keep runtime behavior unchanged.
        return


_load_dotenv_if_exists()

# Base folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
TRANSCRIPT_DIR = os.path.join(DATA_DIR, "transcripts")
INDEX_DIR = os.path.join(DATA_DIR, "indexes")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
SUMMARY_DIR = os.path.join(DATA_DIR, "summaries")
QUIZ_DIR = os.path.join(DATA_DIR, "quizzes")
LECTURE_META_DIR = os.path.join(DATA_DIR, "lectures")
FRAME_DIR = os.path.join(DATA_DIR, "frames")

for d in (
    DATA_DIR,
    TRANSCRIPT_DIR,
    INDEX_DIR,
    UPLOAD_DIR,
    AUDIO_DIR,
    PDF_DIR,
    SUMMARY_DIR,
    QUIZ_DIR,
    LECTURE_META_DIR,
    FRAME_DIR,
):
    os.makedirs(d, exist_ok=True)

# Embedding model for semantic search
# (Good quality + reasonably fast on CPU)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# You could also try "BAAI/bge-small-en-v1.5" if you want better retrieval
# and are okay with a larger download.

# Gemini model configuration
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Approx words per merged caption chunk
MAX_CHUNK_WORDS = 90

# Hybrid retrieval: weight between semantic (cosine) and lexical (BM25)
# 1.0 = only embeddings, 0.0 = only BM25
HYBRID_ALPHA = 0.65

# Retrieval hyperparameters
RETRIEVAL_CANDIDATES = 40   # initial pool size from hybrid ranking
RETRIEVAL_TOPK = 5          # number of chunks we pass to the LLM
NEIGHBOR_WINDOW = 1         # neighbor chunks around each chosen one

# Upload settings
MAX_UPLOAD_MB = 500
MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_UPLOAD_EXTENSIONS = {"mp4", "mkv", "avi", "mov"}

# OCR / frame analysis settings
OCR_FRAME_SAMPLE_SECONDS = 20
OCR_MAX_FRAMES = 60
OCR_MIN_TEXT_CHARS = 24
OCR_MAX_WORDS_PER_FRAME = 80
OCR_MIN_CONFIDENCE = 45.0

# Speechmatics settings
SPEECHMATICS_API_KEY = os.getenv("SPEECHMATICS_API_KEY", "")
SPEECHMATICS_API_URL = os.getenv("SPEECHMATICS_API_URL", "https://asr.api.speechmatics.com/v2")

# Learning assets defaults
SUMMARY_MAX_WORDS = 220
QUIZ_NUM_QUESTIONS = 10
