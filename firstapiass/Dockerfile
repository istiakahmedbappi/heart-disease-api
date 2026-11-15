FROM python:3.12-slim

WORKDIR /app

# Install dependencies first. Docker caches this step, so changing app code won't reinstall them.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code and the trained model
COPY app/ ./app/
COPY model/ ./model/

EXPOSE 8000

# Render gives us a PORT variable; on your computer it falls back to 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]