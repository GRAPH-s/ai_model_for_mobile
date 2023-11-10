FROM pytorch/pytorch:latest

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git wget libgl1-mesa-glx libglib2.0-0 curl gnupg
RUN python3 -m pip install --upgrade pip

WORKDIR /app
COPY . /app
RUN pip install -r /app/requirements.txt
RUN pip install --upgrade translators
RUN pip install timm

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]