import torch
import open_clip
import os
from PIL import Image, ImageOps
import requests
from io import BytesIO

# Global model cache
yolo_model = None
clip_model = None
clip_preprocess = None
device = "cpu"

def setup_models():
    global yolo_model, clip_model, clip_preprocess, device

    yolopath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../yolov5'))
    yolo_model = torch.hub.load(yolopath, 'yolov5s', source='local', trust_repo=True)

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='openai'
    )
    clip_model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model.to(device)

def load_image(path_or_url, is_local=False):
    if is_local:
        return Image.open(path_or_url).convert("RGB")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.114 Safari/537.36"
        )
    }
    res = requests.get(path_or_url, headers=headers, timeout=10)
    res.raise_for_status()
    return Image.open(BytesIO(res.content)).convert("RGB")

def detect_and_crop(image: Image.Image) -> Image.Image:
    results = yolo_model(image)
    boxes = results.xyxy[0]

    if boxes.shape[0] == 0:
        return ImageOps.pad(image, (224, 224), color=(0, 0, 0))

    best_box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    x1, y1, x2, y2 = map(int, best_box[:4])
    cropped = image.crop((x1, y1, x2, y2))

    padded = ImageOps.pad(cropped, (224, 224), color=(0, 0, 0))
    return padded

def normalize_image(img: Image.Image) -> Image.Image:
    return ImageOps.autocontrast(img, cutoff=5)

def standardize_image(image: Image.Image, size=(224, 224)) -> Image.Image:
    return ImageOps.pad(image, size, method=Image.BICUBIC, color=(0, 0, 0), centering=(0.5, 0.5))

def compare_image_by_url(local_path, url2, threshold=0.7):
    global clip_model, clip_preprocess, device

    image1 = load_image(local_path, is_local=True)
    image2 = load_image(url2)

    crop2 = detect_and_crop(image2)

    image1 = standardize_image(normalize_image(image1))
    crop2 = standardize_image(normalize_image(crop2))

    tensor1 = clip_preprocess(image1).unsqueeze(0).to(device)
    tensor2 = clip_preprocess(crop2).unsqueeze(0).to(device)

    with torch.no_grad():
        emb1 = clip_model.encode_image(tensor1)
        emb2 = clip_model.encode_image(tensor2)

    similarity = torch.nn.functional.cosine_similarity(emb1, emb2).item()
    score = round(similarity, 4)

    if score > threshold:
        return {"match": True, "score": score}
    elif score > 0.6:
        return {"match": "maybe", "score": score}
    else:
        return {"match": False, "score": score}
