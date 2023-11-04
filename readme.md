# Проект по генерации описание по фото

1. Установить драйвера дл CUDA на хосте
```bash
distribution=$(. /etc/os-release;echo  $ID$VERSION_ID)  
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -  
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

2. Необходимо создать файл `.env` и присвоить туда ключ в base64 на [сайте](https://developers.sber.ru/studio/workspaces/my-space/get/gigachat-api).
И прописать какую видеокарту хотите использовать.
```bash
echo 'GIGA_CREDENTIALS = "YOUR_KEY" ' > .env
echo 'CUDA = "cuda:1" ' > .env
```
3. Запуск докера. Назову образ `fastapi-app`, а контейнер `mycontainer`
```bash
docker build -t fastapi-app .
```
```bash
docker run -d --gpus all --rm --name mycontainer -p 8800:80 fastapi-app
```
