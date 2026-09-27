FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY talgin ./talgin
COPY demo ./demo

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "talgin.server:app", "--host", "0.0.0.0", "--port", "8000"]
