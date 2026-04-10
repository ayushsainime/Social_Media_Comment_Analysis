# Dockerfile - Phase 7
FROM python:3.11-slim

WORKDIR /app

# Install Reflex first (manages pydantic/sqlmodel compatibility)
RUN pip install --no-cache-dir reflex==0.8.28.post1

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

COPY . .

# Fix Windows CRLF line endings and make executable
RUN sed -i 's/\r$//' start.sh && chmod +x start.sh

EXPOSE 8000
EXPOSE 8001
EXPOSE 3000

CMD ["./start.sh"]