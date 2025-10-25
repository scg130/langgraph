import os

from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

load_dotenv()

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")


def get_chroma_store():
    embedding = OpenAIEmbeddings()
    chroma = Chroma(
        collection_name="docs",
        embedding_function=embedding,
        persist_directory=PERSIST_DIR
    )
    return chroma
