import os

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from pypdf import PdfReader
from tika import parser
from langchain_community.document_loaders import PyPDFDirectoryLoader


def read_pdf_tika(pdf_file_path):
    # Requires Java runtime installed!
    raw = parser.from_file(pdf_file_path)
    pdf_text = raw['content']
    print(pdf_text)
    return pdf_text


def read_pdf_pypdf(pdf_file_path):
    reader = PdfReader(pdf_file_path)
    pdf_text = ""
    for page in reader.pages:
        pdf_text += page.extract_text() + "\n"
    print(pdf_text)
    return pdf_text


def fill_pypdf_loader(context_data_dir):
    pdf_root_path = os.path.join('..', context_data_dir)
    loader = PyPDFDirectoryLoader(pdf_root_path)
    return loader


def read_pdf_dir(pdf_lib='pypdf'):
    pdf_root_path = os.path.join('..', 'pdf')
    pdf_texts = []
    pdf_paths = []
    # Collect all PDF file content
    for file_name in os.listdir(pdf_root_path):
        if not file_name.endswith('.pdf'):
            continue
        pdf_file_path = os.path.join(pdf_root_path, file_name)
        if pdf_lib == 'pypdf':
            pdf_text = read_pdf_pypdf(pdf_file_path)
        elif pdf_lib == 'tika':
            pdf_text = read_pdf_tika(pdf_file_path)
        else:
            raise ValueError("No valid pdf lib defined.")
        pdf_texts.append(pdf_text)
        pdf_paths.append(pdf_file_path)
    return pdf_texts, pdf_paths


def write_pdf_texts():
    pdf_texts, pdf_paths = read_pdf_dir()
    for i, pdf_text in enumerate(pdf_texts):
        with open(pdf_paths[i] + '.txt', 'w', encoding='utf-8') as file:
            file.write(pdf_text)


def create_pdf_db(context_data_dir, embedding_model_name):
    loader = fill_pypdf_loader(context_data_dir)
    docs = loader.load()

    # Naive sanitizing docs before chunking
    for doc in docs:
        doc.page_content = doc.page_content.replace('Produktdatenblatt', '')
        doc.page_content = doc.page_content.replace('__', '')
        doc.page_content = doc.page_content.replace('_', '- ')
        # doc.page_content = doc.page_content.lower()

    # Split row-wise
    splitter = CharacterTextSplitter(separator='\n', chunk_size=0, chunk_overlap=0)
    chunked_docs = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    db = FAISS.from_documents(chunked_docs, embeddings)
    # retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return db


if __name__ == '__main__':
    write_pdf_texts()
