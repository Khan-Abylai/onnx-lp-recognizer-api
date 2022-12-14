FROM python:3.7
ENV TZ=Asia/Almaty
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN export LC_ALL=C.UTF-8
RUN export LANG=C.UTF-8
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

RUN DEBIAN_FRONTEND="noninteractive" apt-get update && apt-get -y install tzdata sudo

RUN apt-get update \
  && apt-get install -y ssh \
    build-essential \
    gcc \
    g++ \
    gdb \
    clang \
    make \
    ninja-build \
    cmake \
    autoconf \
    automake \
    locales-all \
    dos2unix \
    rsync \
    tar \
    python \
    python3-dev \
    python3-pip \
    ffmpeg libsm6 libxext6 libavcodec-dev libavformat-dev libswscale-dev \
    libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev libgtk-3-dev\
    libpng-dev libopenexr-dev libtiff-dev libwebp-dev \
  && apt-get clean

RUN sudo pip3 install tqdm llvmlite scipy opencv-python uvicorn fastapi python-multipart scikit-image pycocotools shortuuid onnxruntime onnx torch

ADD src/ /opt/api/
ADD weights/ /opt/api
WORKDIR /opt/api/
RUN mkdir /opt/api/debug

ENTRYPOINT ["sh", "entrypoint.sh"]
