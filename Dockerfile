FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive
# Make sure all sources are up to date
RUN apt-get update
# Install python dependecies
RUN apt install -y python3 python3-pip
# Install pytest for python exercises
RUN pip3 install -U pytest
RUN pip install --upgrade pip
RUN mkdir ws
#RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
#RUN pip3 install opencv-python
COPY requirements.txt .
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install -r requirements.txt
