import numpy as np
import torch
import requests
from torchvision.models import RegNet_Y_128GF_Weights, regnet_y_128gf
from io import BytesIO
import cv2
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import os


class Sam:
    def __init__(self,
                 model_path="sam_vit_h_4b8939.pth",
                 model_type="vit_h",
                 cuda="cuda:0",
                 accuracy_threshold=0.85):
        self.device = torch.device(cuda if torch.cuda.is_available() else "cpu")
        sam = sam_model_registry[model_type](checkpoint=model_path).to(self.device)
        self.accuracy_threshold = accuracy_threshold
        self.mask_generator = SamAutomaticMaskGenerator(sam)
        self.weights = RegNet_Y_128GF_Weights.DEFAULT
        self.model = regnet_y_128gf(weights=self.weights).to(self.device).eval()
        self.transform = self.weights.transforms()

    @staticmethod
    def download_image(image_url: str):
        try:
            response = requests.get(image_url)
            image_pil = Image.open(BytesIO(response.content))
            image_np = np.array(image_pil, dtype=np.uint8)
            image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            return image_cv2
        except Exception as err:
            print("ошибка при загрузке изображения:", err)
            raise err

    def get_list_of_objects(self, image):
        masks = self.mask_generator.generate(image)
        masks = sorted(masks, key=(lambda x: x['area']), reverse=True)

        current_image = [image]
        for mask in masks:
            mask = mask['segmentation']
            height, width = mask.shape
            object_image = np.zeros((height, width), dtype=np.uint8)
            object_image[mask] = 255
            object_only = cv2.bitwise_and(image, image, mask=object_image)
            current_image.append(object_only)

        category_names = {}
        for img in current_image[:20]:
            input_image = Image.fromarray(img)
            input_tensor = self.transform(input_image).unsqueeze(0)
            with torch.no_grad():
                prediction = self.model(input_tensor.to(self.device)).squeeze(0).softmax(0)

            class_id = torch.argmax(prediction).item()
            score = prediction[class_id].item()
            category_name = self.weights.meta["categories"][class_id]
            category_names[category_name] = score
            # category_names.add(category_name)

        filtered_categories = {category_name: score
                               for category_name, score in category_names.items()
                               if score >= self.accuracy_threshold}
        return set(filtered_categories.keys())

