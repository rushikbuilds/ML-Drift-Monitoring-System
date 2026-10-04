FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY . .

EXPOSE 8000 8501

CMD ["python", "main.py", "--serve", "--host", "0.0.0.0", "--port", "8000"]
