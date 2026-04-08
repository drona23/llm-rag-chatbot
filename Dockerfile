# AWS Lambda container image for Student Loan RAG Chatbot
#
# Build:  docker build -t rag-chatbot .
# Test:   docker run -p 9000:8080 --env-file .env rag-chatbot
# Invoke: curl -X POST http://localhost:9000/2015-03-31/functions/function/invocations \
#           -d '{"httpMethod":"POST","body":"{\"message\":\"What loans are available?\"}"}'

FROM public.ecr.aws/lambda/python:3.11

# Install dependencies first (cached layer -- only rebuilds if requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir "numpy==1.26.4" && \
    pip install --no-cache-dir anthropic pinecone voyageai python-dotenv

# Copy application source
COPY src/ ./src/
COPY lambda_handler.py .

# Lambda handler: file.function_name
CMD ["lambda_handler.handler"]
