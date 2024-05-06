FROM nvcr.io/nvidia/pytorch:22.10-py3

RUN apt-get update && apt-get install -y python3 python3-pip python3-venv python3-wheel

WORKDIR /root

RUN apt update
RUN apt-get update
RUN apt-get install -y nano
RUN apt-get install -y git
RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

RUN pip install notebook numpy librosa pandas matplotlib scipy scikit-learn pytorch_lightning opencv-python albumentations
RUN apt install python-is-python3
RUN apt-get install unzip
RUN python -m pip install ipykernel
RUN python -m ipykernel install --user
RUN git config --global --add safe.directory /workspace/aimsbirdclef
