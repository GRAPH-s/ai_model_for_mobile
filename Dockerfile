FROM pytorch/pytorch:latest

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git python3-pip python3-dev python3-opencv wget libgl1-mesa-glx libglib2.0-0 curl gnupg
RUN python3 -m pip install --upgrade pip
RUN wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda-repo-debian11-12-3-local_12.3.0-545.23.06-1_amd64.deb
RUN dpkg -i cuda-repo-debian11-12-3-local_12.3.0-545.23.06-1_amd64.deb
RUN cp /var/cuda-repo-debian11-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/
RUN add-apt-repository contrib
RUN apt-get update
RUN apt-get -y install cuda-toolkit-12-3
RUN apt-get install -y cuda-drivers

WORKDIR /app
COPY . /app
RUN pip install -r /app/requirements.txt
RUN wget -P /app https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
RUN mkdir -p "/root/.cache/torch/hub/checkpoints/"
RUN wget -P /root/.cache/torch/hub/checkpoints/ https://download.pytorch.org/models/regnet_y_128gf_swag-c8ce3e52.pth

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]