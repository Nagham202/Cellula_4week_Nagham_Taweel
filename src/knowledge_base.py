import json
import random
import shutil

from datasets import load_dataset
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from src.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    HELD_OUT_COUNT,
    HELD_OUT_PATH,
    RANDOM_SEED,
)
from src.embeddings import get_embeddings


def load_humaneval():
    dataset = load_dataset("openai/openai_humaneval", split="test")
    return list(dataset)


def split_held_out(problems):
    shuffled = problems.copy()
    random.Random(RANDOM_SEED).shuffle(shuffled)
    held_out = shuffled[:HELD_OUT_COUNT]
    knowledge = shuffled[HELD_OUT_COUNT:]
    return knowledge, held_out


def problem_to_document(problem):
    full_function = problem["prompt"] + problem["canonical_solution"]
    # A plain-English header helps searches like "write a function that..." find the right problem
    header = f"# Python solution: {problem['entry_point']} ({problem['task_id']})\n"
    metadata = {
        "doc_id": problem["task_id"],
        "source": "humaneval",
        "task_id": problem["task_id"],
        "entry_point": problem["entry_point"],
        "language": "python",
    }
    return Document(page_content=header + full_function, metadata=metadata)


def split_documents(documents):
    # Python-aware: it cuts at def/class lines first, so a function isn't split in the middle
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    # Each chunk gets a fixed id (like HumanEval/12#0), so saving the same thing twice replaces it instead of duplicating it
    counters = {}
    for chunk in chunks:
        doc_name = chunk.metadata["doc_id"]
        number = counters.get(doc_name, 0)
        counters[doc_name] = number + 1
        chunk.metadata["chunk_id"] = f"{doc_name}#{number}"
    return chunks


def open_vector_store():
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_configuration={"hnsw": {"space": "cosine"}},
    )


def load_vector_store():
    # Without this check Chroma quietly creates an empty database and every search returns nothing
    if not CHROMA_DIR.exists():
        raise FileNotFoundError("No knowledge base found. Run: python -m scripts.build_knowledge_base")
    return open_vector_store()


def add_documents(vector_store, documents):
    chunks = split_documents(documents)
    ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    vector_store.add_documents(chunks, ids=ids)
    return len(chunks)


def build_knowledge_base():
    # Start from scratch, otherwise rebuilding would add every problem a second time
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    problems = load_humaneval()
    knowledge, held_out = split_held_out(problems)

    # Keep the held-out problems (with their tests) for the pass@1 evaluation
    HELD_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    HELD_OUT_PATH.write_text(json.dumps(held_out, indent=2), encoding="utf-8")

    documents = [problem_to_document(problem) for problem in knowledge]
    vector_store = open_vector_store()
    chunk_count = add_documents(vector_store, documents)
    return len(problems), len(knowledge), len(held_out), chunk_count
