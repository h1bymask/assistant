FROM nvidia/cuda:11.6.0-base-ubuntu18.04
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update && apt-get install -y git && \
    apt install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt install -y python3.8 && apt install -y python3-pip && \
    pip3 install virtualenv && \
    virtualenv -p python3.8 --no-download venv
COPY requirements.txt ./requirements.txt
SHELL ["/bin/bash", "-c"]
RUN source /venv/bin/activate && \
    pip install -r requirements.txt && \
    pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116 && \
    apt install -y libsndfile1 && pip install numba==0.48

WORKDIR /workspace
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
ENV  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
CMD ["/venv/bin/python","train.py"]
