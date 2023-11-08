from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from BLIP.models.blip import blip_decoder


def load_image(img_url, image_size=384, device="cpu"):
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image


def generate_description(image, model_path="BLIP/model_base_capfilt_large.pth"):
    model_url = model_path
    model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        # beam search
        caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)
        # nucleus sampling
        # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)
        print('caption: ' + caption[0])


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
url = "https://gorodprima.ru/wp-content/uploads/2022/10/4.jpg"
image = load_image(img_url=url,image_size=image_size, device=device)
generate_description(image, model_path)
