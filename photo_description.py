from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip import blip_decoder
import translators as ts
import logging

logging.basicConfig(level=logging.INFO)


class BLIP:
    def __init__(self, model_path="model_base_capfilt_large.pth", image_size=384, device="cpu"):
        logging.info(f"Доступна ли видеокарта? Ответ: {torch.cuda.is_available()}")
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        logging.info(f"Началась загрузка BLIP")
        self.model = blip_decoder(pretrained=model_path, image_size=image_size, vit='base').to(self.device)
        self.model.eval()
        logging.info(f"закончилась загрузка BLIP")

    def generate_description(self, img_url, translator="google"):
        logging.info("Началась загрузка изображения")
        try:
            raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
        except Exception as err:
            logging.info(f"Ошибка при загрузке изображения: {err}")
            return ["Ошибка при загрузке изображения. Недействительный URL или изображение не доступно."]
        logging.info(f"Началась генерация описания")
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        image = transform(raw_image).unsqueeze(0).to(self.device)
        logging.info("Закончилась загрузка изображения")
        with torch.no_grad():
            caption = self.model.generate(image, sample=True, num_beams=3, max_length=50, min_length=10)
        logging.info(f"Закончилась генерация описания")

        caption_rus = ts.translate_text(caption[0], from_language="en", to_language='ru', translator=translator)
        return caption_rus
