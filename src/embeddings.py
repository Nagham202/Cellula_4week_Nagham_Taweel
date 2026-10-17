from langchain_ollama import OllamaEmbeddings

from src.config import EMBEDDING_MODEL


# nomic-embed-text was trained with these prefixes, and search works noticeably better with them
class NomicEmbeddings(OllamaEmbeddings):
    def embed_documents(self, texts):
        labeled = ["search_document: " + text for text in texts]
        return super().embed_documents(labeled)

    def embed_query(self, text):
        return super().embed_query("search_query: " + text)


def get_embeddings():
    return NomicEmbeddings(model=EMBEDDING_MODEL)
