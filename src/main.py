import os
import time
from getpass import getpass
from langchain_community.document_loaders import GitHubIssuesLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSeq2SeqLM
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser
from llama_cpp import Llama

import prepare_context_data
import test


class Timer:
    def __init__(self, process_name):
        self._start_time = None
        self.process_name = process_name

    def __enter__(self):
        if self._start_time is not None:
            raise ValueError
        self._start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._start_time is None:
            raise ValueError
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print(f"{self.process_name} - Laufzeit: {elapsed_time:0.4f} Sekunden")


def run_rag():
    # model_name = 'llama-2-13b-german-assistant-v4.Q2_K.gguf'
    # model_name = 'llama-2-13b-german-assistant-v4.Q3_K_L.gguf'
    model_name = 'llama-2-13b-german-assistant-v4.Q8_0.gguf'
    model_path = os.path.join('..', 'model', model_name)

    context_data_dir = 'pdf'
    # embedding_model_name = 'danielheinz/e5-base-sts-en-de'
    embedding_model_name = 'paraphrase-multilingual-MiniLM-L12-v2'

    model = Llama(model_path=model_path)

    vector_store = prepare_context_data.create_pdf_db(context_data_dir, embedding_model_name)
    test.test_base_query_similarities(vector_store)

    user_query = "Welche Leuchte hat SCIP Nummer dd2ddf15-037b-4473-8156-97498e721fb3?"

    # prompt_template = """[INST] <<SYS>>
    # {system_message}
    # <</SYS>>
    # {user_message} [/INST]"""

    output = ask(model, vector_store, user_query)
    print(output)

    while True:
        input_user_query = input("Frage: ")
        if input_user_query == 'q':
            break
        output = ask(model, vector_store, input_user_query)
        print(output)


def ask(model, db, user_query):
    max_tokens = 200

    with Timer("Retrieval"):
        relevant_docs = db.similarity_search(user_query)

    # TODO Maybe use the most relevant X contents?
    relevant_doc = relevant_docs[0]
    # Add product name manually extracted from file name
    # Expect file name to contain product name after two _
    file_name = os.path.splitext(os.path.basename(relevant_doc.metadata['source']))[0]
    product_name = " ".join(file_name.split('_')[2:])
    context = 'Produktdatenblatt {}: {}'.format(product_name, relevant_doc.page_content)

    prompt_template = "{user_message} Nutze ausschließlich den folgenden Kontext für deine Antwort: \"{context}\""
    prompt = prompt_template.format(user_message=user_query, context=context)
    print("Query-Format: \n{}".format(prompt))

    with Timer("Inferenz"):
        output = model(prompt, max_tokens=max_tokens, echo=False)
    print("Antwort: {}".format(output['choices'][0]['text']))
    return output


if __name__ == '__main__':
    run_rag()
