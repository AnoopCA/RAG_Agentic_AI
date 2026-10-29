import os
import uuid
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_classic.prompts import PromptTemplate
from langchain_classic.schema import HumanMessage

def process_all_PDFs(PDF_Directory):
    all_documents = []
    pdf_path = Path(PDF_Directory)
    pdf_files = list(pdf_path.glob("**/*.pdf"))

    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        documents = loader.load()
        for doc in documents:
            doc.metadata['source_file'] = pdf_file.name
            doc.metadata['file_type'] = "pdf"
        all_documents.extend(documents)

    print(f"Total documents loaded:{(len(all_documents))}")
    return all_documents

def split_documents(documents, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len,
        separators = ["\\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    return split_docs
