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
    faces_list = []
    # Detect faces in the image
    faces = detector.detect_faces(np.array(image))

    # Save faces (cropped images)
    for i, face in enumerate(faces):
        x, y, width, height = face['box']
        
        size = 224, 224
        # Discard crops that are too small (<75 x 75)
        if width >= 75 and height >= 75:
            cropped_img = image.crop([x, y, x + width, y + height])
            # Create thumbnails of crops that are too big
            if width >= 250 and height >= 250:
                cropped_img.thumbnail(size)
            cropped_img.save(f'{destiny_path}/{image_name}')
            faces_list.append(cropped_img)
    
    return faces_list


# Given a list of images, generate vector embeddings and save them to DB
def gen_embeddings(files: list[dict],
                   processor: ViTImageProcessor,
                   device: torch.device,
                   model: ViTModel):
    for img_data in files:
        inputs = processor(images = img_data['img_obj'], return_tensors = "pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs).last_hidden_state.mean(dim = 1).cpu().numpy()
            # embeddings.append(outputs)
            img_data['img_embedding'] = outputs.flatten().tolist()

    return files


# Crop faces and generate embeddings to search 
def gen_embedding_img_to_search(image_path: str,
                         processor: ViTImageProcessor,
                         device: torch.device,
                         model: ViTModel,
                         detector: MTCNN):
    embeddings_to_search = []
    searched_faces = []
    img = Image.open(image_path)

    # Detect faces in the image
    faces = detector.detect_faces(np.array(img))

    # Save faces (cropped images)
    for face in faces:
        x, y, width, height = face['box']
        cropped_img = img.crop([x, y, x + width, y + height])
        searched_faces.append(cropped_img)

        with torch.no_grad():
            inputs = processor(images = cropped_img, return_tensors="pt").to(device)
            img_to_search = model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy().flatten().tolist()
            embeddings_to_search.append(img_to_search)

    return embeddings_to_search, searched_faces