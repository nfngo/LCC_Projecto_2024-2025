from transformers import ViTImageProcessor, ViTModel
from mtcnn import MTCNN
import torch
import numpy as np
from PIL import Image

# Initialize models to process images and face detection
def init_models(model_name: str = "base_patch16_224"):
    # Load ViT model
    device = torch.device("cpu")
    
    if model_name == "dino_vits16":
        # Model: facebook/dino-vits16
        processor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')
        model = ViTModel.from_pretrained('facebook/dino-vits16').to(device)
    else:
        # Model: Vit Base Patch16 224 In21k
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

    # Load MTCNN
    # Create a detector instance
    detector = MTCNN()

    return device, processor, model, detector


# Crop faces and save each of them as a new file
def process_image(image: Image.Image,
                  detector: MTCNN,
                  destiny_path: str,
                  image_name: str):
    
    # # Load an image
    # image = Image.open(image_path)

    # Detect faces in the image
    faces = detector.detect_faces(np.array(image))

    # Save faces (cropped images)
    for i, face in enumerate(faces):
        x, y, width, height = face['box']
        
        cropped_img = image.crop([x, y, x + width, y + height])
        cropped_img.save(f'{destiny_path}/{image_name}.jpg')


# Generate embeddings of all images to save to DB
def gen_embeddings(files: list,
                   processor: ViTImageProcessor,
                   device: torch.device,
                   model: ViTModel):
    embeddings = []
    for img in files:
        inputs = processor(images = img, return_tensors = "pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs).last_hidden_state.mean(dim = 1).cpu().numpy()
            embeddings.append(outputs)

    return np.array(embeddings)


# Crop faces and generate embeddings to search 
def gen_embedding_img_to_search(image_path: str,
                         processor: ViTImageProcessor,
                         device: torch.device,
                         model: ViTModel,
                         detector: MTCNN):
    embeddings_to_search = []
    img = Image.open(image_path)

    # Detect faces in the image
    faces = detector.detect_faces(np.array(img))

    # Save faces (cropped images)
    for face in faces:
        x, y, width, height = face['box']
        cropped_img = img.crop([x, y, x + width, y + height])

        inputs = processor(images = cropped_img, return_tensors="pt").to(device)
        img_to_search = model(**inputs).last_hidden_state.mean(dim=1)[0].tolist()
        embeddings_to_search.append(img_to_search)

    return embeddings_to_search