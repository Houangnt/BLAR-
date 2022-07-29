FROM python:3.8.12

RUN apt-get update && apt-get install curl ffmpeg libsm6 libxext6 ssh git locales -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app-src

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

ENV TZ=Asia/Ho_Chi_Minh
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

COPY . .

CMD ["python", "pipeline.py"]