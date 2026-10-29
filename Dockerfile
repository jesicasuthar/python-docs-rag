FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py rag.py logger.py ./

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=7860", "--server.headless=true"]
