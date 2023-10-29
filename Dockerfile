FROM pytorch/pytorch:latest

WORKDIR /app

COPY ./ /app
COPY requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt
RUN wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]