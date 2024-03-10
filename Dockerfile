FROM paddlepaddle/paddle:2.4.1-gpu-cuda11.7-cudnn8.4-trt8.4

RUN apt-get update -y
RUN apt-get install python3.10 -y
RUN apt-get install python3-pip -y
RUN python3.8 -m pip install --upgrade pip
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y
RUN apt-get install python3.10-dev -y

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

WORKDIR /app-src

RUN python3.8 -m pip install paddlepaddle-gpu==2.4.1.post117 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

COPY . /app-src/

RUN python3.10 -m pip install -r requirements.txt
