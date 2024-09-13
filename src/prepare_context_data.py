import os

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
    pdf_root_path = os.path.join('..', 'pdf')
    for i, pdf_text in enumerate(pdf_texts):
        with open(pdf_paths[i] + '.txt', 'w', encoding='utf-8') as file:
            file.write(pdf_text)


def test():
    pdf_root_path = os.path.join('..', 'pdf')
    # pdf_file_name = 'ZMP_55877_XBO_4000_W_HSA_OFR.pdf'
    pdf_file_name = 'ZMP_55864_XBO_2000_W_SHSC_OFR.pdf'
    pdf_file_path = os.path.join(pdf_root_path, pdf_file_name)
    read_pdf_tika(pdf_file_path)


def create_pdf_db(context_data_dir, embedding_model_name):
    loader = fill_pypdf_loader(context_data_dir)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunked_docs = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    db = FAISS.from_documents(chunked_docs, embeddings)
    # retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    ###
    # Alternative with FAISS:
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    vector_store.add_documents(documents=docs)

    # test_base_query_similarities(vector_store)

    return db


def test_base_query_similarities(vector_store: FAISS):
    # English version
    labeled_queries = [
        ("How much does XBO 4000 W/HS XL OFR weigh?", 'ZMP_1007199_XBO_4000_W_HS_XL_OFR.pdf'),
        ("Which product has SCIP number dd2ddf15-037b-4473-8156-97498e721fb3?", 'ZMP_1007193_XBO_3000_W_HS_XL_OFR.pdf'),
        ("Which product has article identifier 4008321299963", 'ZMP_1007189_XBO_2500_W_HS_XL_OFR.pdf'),
    ]
    for query, relevant_doc in labeled_queries:
        results = vector_store.similarity_search(query)
        best_result = results[0]
        source_path = best_result.metadata['source']
        source_file = os.path.basename(source_path)
        hit = source_file == relevant_doc
        prefix = '[{}]: '.format('x' if hit else ' ')
        print(prefix + query)
        print(results)


def test_faiss():
    # https://python.langchain.com/v0.2/docs/integrations/vectorstores/faiss/
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    from uuid import uuid4

    from langchain_core.documents import Document

    document_1 = Document(
        page_content="I had chocalate chip pancakes and scrambled eggs for breakfast this morning.",
        metadata={"source": "tweet"},
    )

    document_2 = Document(
        page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
        metadata={"source": "news"},
    )

    document_3 = Document(
        page_content="Building an exciting new project with LangChain - come check it out!",
        metadata={"source": "tweet"},
    )

    document_4 = Document(
        page_content="Robbers broke into the city bank and stole $1 million in cash.",
        metadata={"source": "news"},
    )

    document_5 = Document(
        page_content="Wow! That was an amazing movie. I can't wait to see it again.",
        metadata={"source": "tweet"},
    )

    document_6 = Document(
        page_content="Is the new iPhone worth the price? Read this review to find out.",
        metadata={"source": "website"},
    )

    document_7 = Document(
        page_content="The top 10 soccer players in the world right now.",
        metadata={"source": "website"},
    )

    document_8 = Document(
        page_content="LangGraph is the best framework for building stateful, agentic applications!",
        metadata={"source": "tweet"},
    )

    document_9 = Document(
        page_content="The stock market is down 500 points today due to fears of a recession.",
        metadata={"source": "news"},
    )

    document_10 = Document(
        page_content="I have a bad feeling I am going to get deleted :(",
        metadata={"source": "tweet"},
    )

    documents = [
        document_1,
        document_2,
        document_3,
        document_4,
        document_5,
        document_6,
        document_7,
        document_8,
        document_9,
        document_10,
    ]
    uuids = [str(uuid4()) for _ in range(len(documents))]

    vector_store.add_documents(documents=documents, ids=uuids)

    results = vector_store.similarity_search(
        "LangChain provides abstractions to make working with LLMs easy",
        k=2,
        filter={"source": "tweet"},
    )
    for res in results:
        print(f"* {res.page_content} [{res.metadata}]")


if __name__ == '__main__':
    test_faiss()
