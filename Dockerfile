FROM pytorch/pytorch:latest

WORKDIR /app

COPY ./ /app
COPY requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]