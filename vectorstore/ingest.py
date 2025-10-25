from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def ingest_file(text: str, metadata: dict = None):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = splitter.create_documents([text], metadatas=[metadata])

    embedding = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
    vectorstore = Chroma(collection_name="docs", embedding_function=embedding, persist_directory="./chroma_store")
    vectorstore.add_documents(docs)
    vectorstore.persist()

    return {"status": "ok", "chunks": len(docs)}