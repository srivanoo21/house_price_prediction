FROM python:3.8-slim-buster
WORKDIR /app
COPY . /app

RUN apt update -y

RUN pip install -r requirements.txt
CMD ["python", "app.py"]