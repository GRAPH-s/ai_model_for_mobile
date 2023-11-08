from dotenv import load_dotenv
import os
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

"""
Тебе необходимо от первого лица создать креативное описание на русском языке к фотографии, 
чтобы потом выставить это описание в социальную сеть. Описание должно быть стильным, модным, молодежным. 
Я тебе передаю список объектов на фото. Если слова будут на другом языке, то переведи их на русский. 
Добавь 5 хэштегов на английском языке, чтобы повыстить популярность поста. Используй эмоджи. 
Хэштеги придумай исходя из контекста описания.
"""

load_dotenv(".env")
GIGA_CREDENTIALS = os.getenv("GIGA_CREDENTIALS")


class Description:
    def __init__(self, temperature=0.7, max_tokens=300):
        self.temperature = temperature
        self.max_tokens = max_tokens

    def get_description(self, instruction: str, objects: str):
        payload = Chat(
            messages=[
                Messages(
                    role=MessagesRole.SYSTEM,
                    content=instruction
                )
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        giga = GigaChat(credentials=GIGA_CREDENTIALS, scope="GIGACHAT_API_PERS", temperature=1.4, verify_ssl_certs=False)
        user_input = objects
        payload.messages.append(Messages(role=MessagesRole.USER, content=user_input))
        response = giga.chat(payload)
        payload.messages.append(response.choices[0].message)
        return response.choices[0].message.content
