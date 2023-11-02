FROM nvidia/cuda:11.4.0-base-ubuntu20.04 as builder

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install -y git wget libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

FROM pytorch/pytorch:latest

COPY --from=builder / /
WORKDIR /app
COPY . /app

RUN pip install -r /app/requirements.txt
RUN wget -P /app https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
RUN mkdir -p "/root/.cache/torch/hub/checkpoints/"
RUN wget -P /root/.cache/torch/hub/checkpoints/ https://download.pytorch.org/models/regnet_y_128gf_swag-c8ce3e52.pth

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]