FROM pytorch/pytorch:latest

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git wget libgl1-mesa-glx libglib2.0-0 curl gnupg
RUN python3 -m pip install --upgrade pip


WORKDIR /app
COPY . /app
RUN pip install -r /app/requirements.txt
RUN wget -P /app https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
RUN mkdir -p "/root/.cache/torch/hub/checkpoints/"
RUN wget -P /root/.cache/torch/hub/checkpoints/ https://download.pytorch.org/models/vit_h_14_swag-80465313.pth

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]