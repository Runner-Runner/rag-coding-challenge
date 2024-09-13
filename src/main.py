import os
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


def run_rag2():
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
    # user_message = "Wie lautet die Hauptstadt von Belgien? Und was sind Wahrzeichen dort?"
    # prompt = prompt_template.format(system_message=system_message, user_message=user_message)
    output = model(prompt, max_tokens=max_tokens, echo=False)
    print(output)

    while True:
        input_user_message = input("Question: ")
        if input_user_message == 'q':
            break
        prompt = prompt_template.format(system_message=system_message, user_message=input_user_message)
        output = model(prompt, max_tokens=max_tokens, echo=False)
        print(output)


def run_rag():
    # import torch
    # cuda_available = torch.cuda.is_available()
    # device_count = torch.cuda.device_count()
    # current_device = torch.cuda.current_device()

    ACCESS_TOKEN = getpass("YOUR_GITHUB_PERSONAL_TOKEN")
    loader = GitHubIssuesLoader(repo="huggingface/peft", access_token=ACCESS_TOKEN, include_prs=False, state="all",
                                page=1)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=30)
    chunked_docs = splitter.split_documents(docs)
    # db = FAISS.from_documents(chunked_docs, HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5"))
    # retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # model_name = 'HuggingFaceH4/zephyr-7b-beta'
    # model_name = 'saadrasheeddev/meta-llama3-8B-instruct-4bit-quantized'
    model_name = 'DiscoResearch/Llama3-German-8B'
    # TOO BAD: model_name = 'lmsys/fastchat-t5-3b-v1.0'

    # Cannot be used with my gpu
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    # model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.2,
        do_sample=True,
        repetition_penalty=1.1,
        return_full_text=True,
        max_new_tokens=400,
        # device=current_device,
    )
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    prompt_template = """
    <|system|>
    Answer the question based on your knowledge. Use the following context to help:
    
    {context}
    
    </s>
    <|user|>
    {question}
    </s>
    <|assistant|>
    
     """
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )
    llm_chain = prompt | llm | StrOutputParser()

    question = "How do you combine multiple adapters?"

    context = ''
    output = ask(llm_chain=llm_chain, question=question, context=context)
    print(output)

    while True:
        input_question = input("Ask a question: ")
        if input_question == 'q':
            break
        output = ask(llm_chain=llm_chain, question=input_question, context=context)
        print(output)

    # retriever = db.as_retriever()
    # rag_chain = {"context": retriever, "question": RunnablePassthrough()} | llm_chain
    # output = rag_chain.invoke(question)
    # print(output)


def ask(llm_chain, question, context=''):
    return llm_chain.invoke({"context": context, "question": question})


if __name__ == '__main__':
    run_rag2()
