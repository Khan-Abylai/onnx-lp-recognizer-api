FROM python:3

ENV TZ=Asia/Almaty
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN DEBIAN_FRONTEND="noninteractive" apt-get update && apt-get -y install \
    tzdata sudo ssh build-essential gcc g++ gdb clang make ninja-build \
    cmake autoconf automake locales-all dos2unix rsync tar python-is-python3 \
    python3-dev python3-pip ffmpeg libsm6 libxext6 iputils-ping && \
    apt-get clean

RUN pip3 install --no-cache-dir \
    Flask matplotlib numpy opencv-python opencv-contrib-python \
    python-telegram-bot==13.15 \
    requests scikit-image scipy pycocotools nltk gunicorn shortuuid

ADD src/ /opt/api/
WORKDIR /opt/api/

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["sh", "entrypoint.sh"]
