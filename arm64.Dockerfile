from mxnet/python:1.9.1_aarch64_cpu_py3

WORKDIR /app

RUN apt-get update && apt-get install -y libssl-dev zlib1g-dev gcc g++ make git wget pkg-config

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

RUN mkdir -p beomi/KcELECTRA-base-v2022 && cd beomi/KcELECTRA-base-v2022 && \ 
    wget https://huggingface.co/beomi/KcELECTRA-base-v2022/raw/main/tokenizer_config.json&& \
    wget https://huggingface.co/beomi/KcELECTRA-base-v2022/raw/main/config.json && \
    wget https://huggingface.co/beomi/KcELECTRA-base-v2022/raw/main/vocab.txt && \
    wget https://huggingface.co/beomi/KcELECTRA-base-v2022/raw/main/special_tokens_map.json && \
    cd ../..

COPY requirements.txt .

RUN pip3 install -r requirements.txt

# RUN pip3 install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'

# RUN pip install quickspacer tensorflow_io==0.29.0


COPY . .

RUN touch Test.py && cat Analyzer.py >> Test.py && cat TestCode.py >> Test.py && rm -f TestCode.py

ENTRYPOINT ["python3"]
