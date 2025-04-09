from transformers import ViTImageProcessor, ViTModel
from qdrant_client import QdrantClient
from qdrant_client.http import models
from PIL import Image
import numpy as np
import torch
import os

import qdrant.utils as qd

# Using containers
# client = QdrantClient(host="qdrant", port=6333)

# Local dev
client = QdrantClient(host="localhost", port=6333)

collection = "image_collection"

# facebook/dino-vits16 size: 384
# Vit Base Patch16 224 In21k size: 768
qd.create_collection(client, collection, 768, models.Distance.DOT)

# Create collection
# if not client.collection_exists(collection_name=collection):
#     client.create_collection(
#         collection_name=collection,
#         vectors_config=models.VectorParams(size=768, distance=models.Distance.DOT)
#     )

image_names = []
image_files = []

current_directory = os.getcwd()
extensions = (".jpg", ".jpeg")

# Save name and img obj from every image in {path}
def get_images_info(path):
    image_names = []
    image_files = []

    for file in os.listdir(f"{path}"):
        if file.endswith(extensions):
            image_names.append(file.split(".")[0])
            image_files.append(Image.open(os.path.join(f"{path}",file)))    
    
    return image_names, image_files

image_names, image_files = get_images_info(f"{current_directory}/assets")



thumbnails_path = "thumbnails"
# Create directory if needed
if not os.path.exists(thumbnails_path):
    os.makedirs(thumbnails_path)
 
# Resize images (performance) and save them in thumbnails directory
size = 224, 224
for i, image in enumerate(image_files):
    image.thumbnail(size)
    image.save(f"{current_directory}/{thumbnails_path}/{image_names[i]}_thumbnail.jpeg")

# Load ViT model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model: facebook/dino-vits16
# processor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')
# model = ViTModel.from_pretrained('facebook/dino-vits16').to(device)

# Model: Vit Base Patch16 224 In21k
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

# Get thumbnails information
thumb_names, thumb_files = get_images_info(f"{current_directory}/{thumbnails_path}")

# Generate embeddings
def get_embeddings(files):
    embeddings = []
    for img in files:
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(outputs)
    return embeddings

gen_embeddings = get_embeddings(thumb_files)
embeddings = np.array(gen_embeddings)
print(embeddings)

# Save embeddings to DB
for i in range(0, len(image_files)):
    client.upsert(
        collection_name = collection,
        points = [models.PointStruct(
            id = i,
            payload = {
                "name": image_names[i]
            },
            vector = embeddings[i][0],
        )]
    )


img = Image.open(f"{current_directory}/Marcelo_Montenegro.jpg")
inputs = processor(images=img, return_tensors="pt").to(device)
img_to_search = model(**inputs).last_hidden_state.mean(dim=1)[0].tolist()

nearest = client.query_points(
    collection_name=collection,
    query=img_to_search,
    limit=5,
    with_payload=True
)

# print(nearest)

import subprocess
def see_images(results, top_k=2):
    for i, result in enumerate(results):
        # image_id = results[i]['id']
        # name    = results[i].payload['name']
        # score = results[i].score
        # image = Image.open(image_files[image_id])
        test = result[1]
        for i in range(len(test)):
            # print(f"{i}: {test[i]}")
            name = test[i].payload['name']
            score = test[i].score
            print(f"Result #{i+1}: {name} was diagnosed with {score * 100} confidence")
            print(f"This image score was {score}")
            print("-" * 50)
            print()

see_images(nearest, top_k=2)