Prerequisites:
- python installed (tested with 3.12)
- Download model: https://huggingface.co/TheBloke/Llama-2-13B-German-Assistant-v4-GGUF/blob/main/llama-2-13b-german-assistant-v4.Q2_K.gguf
- Download pdf zip file

Setup:
- Clone repository from github: https://github.com/Runner-Runner/rag-coding-challenge.git
- cd into repo root
- Copy downloaded model into model/ dir
- Copy unzipped PDF files into pdf/ dir
- Create venv 'venv_cc' and install libraries by running python script create_venv.py in root
- Run main.py with venv-specific python