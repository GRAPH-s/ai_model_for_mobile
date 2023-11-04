from dotenv import load_dotenv
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Union
import chat_description
from detect_object import Sam


class Objects(BaseModel):
    description: Union[str, None] = None
    image_url: Union[str, None] = None
    accuracy_threshold: Union[float, None] = 0.85
    number_objects: Union[int, None] = 5


load_dotenv(".env")
CUDA = os.getenv("CUDA")
app = FastAPI()
sam = Sam(cuda=CUDA)


instruction_1 = """
Тебе необходимо от первого лица создать креативное описание на русском языке к фотографии, 
чтобы потом выставить это описание в социальную сеть. Описание должно быть стильным, модным, молодежным. 
Я тебе передаю список объектов на фото. Если слова будут на другом языке, то переведи их на русский. 
Добавь 5 хэштегов на английском языке, чтобы повысить популярность поста. Используй эмоджи. 
Хэштеги придумай исходя из контекста описания.
"""
chat_1 = chat_description.Description(instruction=instruction_1)


@app.post("/api/without_input/")
def root(objects: Objects):
    image = sam.download_image(objects.image_url)
    if isinstance(image, str):
        return JSONResponse(content={"error": image}, status_code=400)
    objects_on_image = sam.get_list_of_objects(image=image,
                                               accuracy_threshold=objects.accuracy_threshold,
                                               number_objects=objects.number_objects)
    objects_on_image = ", ".join(objects_on_image)
    description = chat_1.get_description(objects_on_image)
    if len(objects_on_image) == 0:
        objects_on_image = f"Не смог ничего обнаружить с точностью: {objects.accuracy_threshold}"
    return JSONResponse(content={"description": description,
                                 "detected objects": objects_on_image},
                        status_code=200)


instruction_2 = """
Тебе необходимо от первого лица продолжить описание на русском языке к фотографии, 
чтобы потом выставить это описание в социальную сеть. Описание должно быть стильным, модным, молодежным. 
Я тебе сначала дам начало описание, а после символа ; передам список объектов на фото. 
Если слова будут на другом языке, то переведи их на русский. 
Добавь 5 хэштегов на английском языке, чтобы повысить популярность поста. Используй эмоджи. 
Хэштеги придумай исходя из контекста описания.
"""
chat_2 = chat_description.Description(instruction=instruction_2)


@app.post("/api/with_input/")
def root(objects: Objects):
    image = sam.download_image(objects.image_url)
    if objects.description is None:
        return JSONResponse(content={"error": "Описание не может быть пустым"}, status_code=400)
    if isinstance(image, str):
        return JSONResponse(content={"error": image}, status_code=400)
    objects_on_image = sam.get_list_of_objects(image=image,
                                               accuracy_threshold=objects.accuracy_threshold,
                                               number_objects=objects.number_objects)
    objects_on_image = ", ".join(objects_on_image)
    beginning = objects.description + ";" + objects_on_image
    description = chat_2.get_description(beginning)
    if len(objects_on_image) == 0:
        objects_on_image = f"Не смог ничего обнаружить с точностью: {objects.accuracy_threshold}"
    return JSONResponse(content={"description": description,
                                 "detected objects": objects_on_image},
                        status_code=200)

