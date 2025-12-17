FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm


COPY . .

RUN chmod +x start.sh

EXPOSE 8000
EXPOSE 8501

CMD ["./start.sh"]
#docker build -t ayushsainime/twitter-reddit-nlp:1.0 .
#docker run -p 8000:8000 -p 8501:8501 ayushsainime/twitter-reddit-nlp:1.0
#docker push ayushsainime/twitter-reddit-nlp:1.0

