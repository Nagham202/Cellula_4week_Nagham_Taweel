import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

# The Groq key lives in .env so it never ends up in the code or on GitHub
load_dotenv(PROJECT_DIR / ".env")

# Switch to "ollama" to run everything locally (slower, but no limits and works offline)
LLM_PROVIDER = "groq"

# llama-3.3-70b-versatile isn't available on my account, so I'm using this one
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

OLLAMA_MODEL = "llama3.1:8b"

# 0 so the classifier and the judge give the same answer every time
LLM_TEMPERATURE = 0

# If you change this, rebuild the database (old vectors won't match the new model)
EMBEDDING_MODEL = "nomic-embed-text"

CHROMA_DIR = DATA_DIR / "chroma_db"
COLLECTION_NAME = "coding_knowledge"

HELD_OUT_PATH = DATA_DIR / "held_out_tasks.json"
HELD_OUT_COUNT = 20
# Fixed seed so the same 20 problems are held out every time I rebuild
RANDOM_SEED = 42

# The longest HumanEval problem is about 2000 characters, so every problem stays in one piece
CHUNK_SIZE = 2500
CHUNK_OVERLAP = 200

TOP_K = 3

# Stops infinite loops (and web servers, which never stop on their own)
RUN_TIMEOUT_SECONDS = 10
MAX_OUTPUT_CHARS = 10_000

USER_SOLUTIONS_PATH = DATA_DIR / "user_solutions.jsonl"

MEMORY_PATH = DATA_DIR / "memory.json"
# Everything is saved, but only this much goes to the LLM (free tier = 8,000 tokens a minute)
MEMORY_TURNS_IN_PROMPT = 3
MEMORY_TEXT_LIMIT = 400
MEMORY_CODE_LIMIT = 2000
# Follow-ups are short ("in python", "make it faster"); longer messages explain themselves
MAX_FOLLOW_UP_CHARS = 200
