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
    model_name = 'llama-2-13b-german-assistant-v4.Q2_K.gguf'
    model_path = os.path.join('..', 'model', model_name)

    context_data_dir = 'pdf'
    # embedding_model_name = 'danielheinz/e5-base-sts-en-de'
    embedding_model_name = 'paraphrase-multilingual-MiniLM-L12-v2'

    model = Llama(model_path=model_path)

    db = prepare_context_data.create_pdf_db(context_data_dir, embedding_model_name)

    system_message = ("Du bist ein Assistent, der prägnante, nicht ausschweifende Antworten gibt. "
                      "Hier ist der Kontext für deine Antwort: {context}")
    max_tokens = 200

    user_message = "Welche Leuchte hat SCIP Nummer dd2ddf15-037b-4473-8156-97498e721fb3?"
    relevant_docs = db.similarity_search(user_message)
    context = relevant_docs[0].page_content

    # prompt_template = """[INST] <<SYS>>
    # {system_message}
    # <</SYS>>
    # {user_message} [/INST]"""

    # TODO Probably not the right format yet for this model ...
    prompt_template = '### User: {user_message}\n### Assistant: {system_message}'

    prompt = prompt_template.format(user_message=user_message, system_message=system_message.format(context=context))
    # prompt = prompt_template.format(system_message=system_message, user_message=user_message)
    output = ask(model, prompt, max_tokens)
    print(output)

    while True:
        input_user_message = input("Frage: ")
        if input_user_message == 'q':
            break
        prompt = prompt_template.format(system_message=system_message, user_message=input_user_message)
        output = ask(model, prompt, max_tokens)
        print(output)


def ask(model, prompt, max_tokens):
    with Timer("Inferenz"):
        output = model(prompt, max_tokens=max_tokens, echo=False)
    return output


if __name__ == '__main__':
    run_rag()
