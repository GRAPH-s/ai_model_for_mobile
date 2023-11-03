# Проект по генерации описание по фото

1. Необходимо создать файл `.env` и присвоить туда ключ в base64 на [сайте](https://developers.sber.ru/studio/workspaces/my-space/get/gigachat-api)
```bash
echo 'GIGA_CREDENTIALS = "YOUR_KEY" ' > .env
```
2. Запуск докера. Назову образ `fastapi-app`, а контейнер `mycontainer`
```bash
docker build -t fastapi-app .
```
```bash
docker run -d --gpus all -rm --name mycontainer -p 8800:80 fastapi-app
```
