import os

# Base folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
TRANSCRIPT_DIR = os.path.join(DATA_DIR, "transcripts")
INDEX_DIR = os.path.join(DATA_DIR, "indexes")

for d in (DATA_DIR, TRANSCRIPT_DIR, INDEX_DIR):
    os.makedirs(d, exist_ok=True)

# Embedding model for semantic search
# (Good quality + reasonably fast on CPU)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# You could also try "BAAI/bge-small-en-v1.5" if you want better retrieval
# and are okay with a larger download.

# Approx words per merged caption chunk
MAX_CHUNK_WORDS = 90

# Hybrid retrieval: weight between semantic (cosine) and lexical (BM25)
# 1.0 = only embeddings, 0.0 = only BM25
HYBRID_ALPHA = 0.65

# Retrieval hyperparameters
RETRIEVAL_CANDIDATES = 40   # initial pool size from hybrid ranking
RETRIEVAL_TOPK = 5          # number of chunks we pass to Gemini
NEIGHBOR_WINDOW = 1         # neighbor chunks around each chosen one
