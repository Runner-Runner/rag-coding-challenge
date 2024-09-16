import os
import sys
import time
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


def run_rag(initial_query=None):
    # model_name = 'llama-2-13b-german-assistant-v4.Q2_K.gguf'
    # model_name = 'llama-2-13b-german-assistant-v4.Q3_K_L.gguf'
    # model_name = 'llama-2-13b-german-assistant-v4.Q8_0.gguf'
    # mistral: MUCH better answer quality and 3x as fast. Does not always answer in German
    model_name = 'mistral-7b-instruct-v0.2.Q5_K_M.gguf'
    model_path = os.path.join('..', 'model', model_name)

    prompt_template = ("### User: {user_message} Nutze den folgenden Kontext für deine Antwort: \n"
                       "\"{context}\"\n### Assistant:")
    if 'mistral' in model_name:
        prompt_template = (
            "<s>[INST] {user_message} Antworte auf deutsch. Nutze den folgenden Kontext für deine Antwort: \n"
            "\"{context}\" [/INST]")

    context_data_dir = 'pdf'
    # embedding_model_name = 'danielheinz/e5-base-sts-en-de'
    embedding_model_name = 'paraphrase-multilingual-MiniLM-L12-v2'

    model = Llama(model_path=model_path)

    vector_store = prepare_context_data.create_pdf_db(context_data_dir, embedding_model_name)

    # test.test_extended_simple_queries(vector_store)

    if initial_query:
        output = ask(model, vector_store, initial_query, prompt_template)
        print(output)

    while True:
        input_user_query = input("Frage: ")
        if input_user_query == 'q':
            break
        output = ask(model, vector_store, input_user_query, prompt_template)
        print(output)


def ask(model, db, user_query, prompt_template):
    max_tokens = 300

    use_most_relevant_k = 4
    with Timer("Retrieval"):
        relevant_docs = db.similarity_search(user_query, k=use_most_relevant_k)

    # TODO Maybe use the most relevant X contents?
    # Add product name manually extracted from file name
    # Expect file name to contain product name after two _

    contexts = []
    for i in range(use_most_relevant_k):
        relevant_doc = relevant_docs[i]
        file_name = os.path.splitext(os.path.basename(relevant_doc.metadata['source']))[0]
        product_name = " ".join(file_name.split('_')[2:])
        context = 'Produktdatenblatt {}: {}'.format(product_name, relevant_doc.page_content)
        contexts.append(context)
    full_context = '\n'.join(contexts)

    prompt = prompt_template.format(user_message=user_query, context=full_context)
    print("Query-Format: \n{}".format(prompt))

    with Timer("Inferenz"):
        output = model(prompt, max_tokens=max_tokens, echo=False)
    print("Antwort: {}".format(output['choices'][0]['text']))
    return output


if __name__ == '__main__':
    initial_query = None
    if len(sys.argv) == 2:
        initial_query = sys.argv[1]
    run_rag(initial_query)
