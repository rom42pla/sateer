# Specify the parent image from which we build
FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

RUN apt-get update &&  \
    apt-get install -y wget &&  \
    apt-get clean

COPY requirements.txt .

# Build the application with cmake
ENV CONDA_DIR /opt/conda
RUN wget -O conda.sh --quiet https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh &&  \
    bash conda.sh -b -p /opt/conda &&  \
    rm conda.sh
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda init bash

RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install -r requirements.txt