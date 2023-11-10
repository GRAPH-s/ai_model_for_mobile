from dotenv import load_dotenv
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Union
import chat_description
from detect_object import Sam
from photo_description import BLIP


class Objects(BaseModel):
    description: Union[str, None] = None
    image_url: Union[str, None] = None
    instruction: Union[str, None] = None


load_dotenv(".env")
CUDA = os.getenv("CUDA")
app = FastAPI()
blip = BLIP(device=CUDA)
chat = chat_description.Description()

instruction_1 = """
Создай описание от первого лица по фотографии для социальной сети. 
Описание должно быть стильным, модным и молодежным. Я тебе отправлю информацию о содержании фото. 
Предлагаю добавить несколько хэштегов на английском языке, чтобы сделать пост более популярным. 
Давай ты придумаешь подходящие хэштеги в соответствии с настроением описания.
"""

instruction_2 = """
Создай описание от первого лица по фотографии для социальной сети. 
Описание должно быть стильным, модным и молодежным.
Я тебе сначала дам начало описание, а после символа ; передам информацию о содержании фото. 
Предлагаю добавить несколько хэштегов на английском языке, чтобы сделать пост более популярным. 
Давай ты придумаешь подходящие хэштеги в соответствии с настроением описания.
"""


@app.post("/api/without_input/")
def root(objects: Objects):
    description_image = blip.generate_description(img_url=objects.image_url)
    if isinstance(description_image, list):
        return JSONResponse(content={"error": description_image[0]}, status_code=400)
    instruction = instruction_1 if objects.instruction is None else objects.instruction
    description = chat.get_description(instruction, f"На фото: {description_image}")
    if len(description_image) == 0:
        objects_on_image = f"Не смог ничего обнаружить на фото"
        return JSONResponse(content={"detected objects": objects_on_image}, status_code=500)
    return JSONResponse(content={"description": description,
                                 "description_image_from_BLIP": description_image
                                 },
                        status_code=200)


@app.post("/api/with_input/")
def root(objects: Objects):
    description_image = blip.generate_description(img_url=objects.image_url)
    if isinstance(description_image, list):
        return JSONResponse(content={"error": description_image[0]}, status_code=400)

    beginning = objects.description + ";" + description_image
    instruction = instruction_2 if objects.instruction is None else objects.instruction
    description = chat.get_description(instruction, beginning)
    if len(description_image) == 0:
        objects_on_image = f"Не смог ничего обнаружить на фото"
        return JSONResponse(content={"detected objects": objects_on_image}, status_code=500)
    return JSONResponse(content={"description": description,
                                 "description_image_from_BLIP": description_image
                                 },
                        status_code=200)

