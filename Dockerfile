FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

RUN apt-get update && apt-get install -y python3 python3-pip python3-venv python3-wheel

WORKDIR /root

RUN apt update
RUN apt-get update
RUN apt-get install -y nano
RUN apt-get install -y git
RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install notebook
RUN apt install python-is-python3
RUN apt-get install unzip
RUN python -m pip install ipykernel
RUN python -m ipykernel install --user
RUN git config --global --add safe.directory /workspace/aimsbirdclef
