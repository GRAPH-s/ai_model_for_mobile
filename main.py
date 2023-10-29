from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union
import chat_description
from detect_object import Sam


class Objects(BaseModel):
    description: Union[str, None] = None
    image_url: Union[str, None] = None


app = FastAPI()
sam = Sam()

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
    try:
        image = sam.download_image(objects.image_url)
    except Exception as e:
        return {"error": e}
    objects_on_image = ", ".join( sam.get_list_of_objects(image))
    description = chat_1.get_description(objects_on_image)
    return {"description": description}


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
    try:
        image = sam.download_image(objects.image_url)
    except Exception as e:
        return {"error": e}
    objects_on_image = ", ".join(sam.get_list_of_objects(image))
    beginning = objects.description + ";" + objects_on_image
    description = chat_2.get_description(beginning)
    return {"description": description}

