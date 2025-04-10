from transformers import ViTImageProcessor, ViTModel
from qdrant_client import QdrantClient
from qdrant_client.http import models
from PIL import Image
import numpy as np
import torch
import os

import qdrant.utils as qd
import models.models as mdl

# Using containers
# client = QdrantClient(host="qdrant", port=6333)

# Local dev
client = QdrantClient(host="localhost", port=6333)

collection = "image_collection"

# facebook/dino-vits16 size: 384
# Vit Base Patch16 224 In21k size: 768
qd.create_collection(client, collection, 768, models.Distance.COSINE)

image_names = []
image_files = []

current_directory = os.getcwd()
extensions = (".jpg", ".jpeg")

# Save name and img obj from every image in {path}
def get_images_info(path):
    img_names = []
    img_files = []

    for file in os.listdir(f"{path}"):
        if file.endswith(extensions):
            img_names.append(file.split(".")[0])
            img_files.append(Image.open(os.path.join(f"{path}",file)))    
    
    return img_names, img_files

image_names, image_files = get_images_info(f"{current_directory}/assets")

# Create directory if needed
faces_path = "faces"
if not os.path.exists(faces_path):
    os.makedirs(faces_path)
 
device, processor, model, detector = mdl.init_models()

# Resize images (performance) and save them in thumbnails directory
# size = 224, 224
# for i, image in enumerate(image_files):
#     image.thumbnail(size)
#     image.save(f"{current_directory}/{thumbnails_path}/{image_names[i]}_thumbnail.jpeg")

for i, image in enumerate(image_files):
    destiny_path = f"{current_directory}/{faces_path}/"
    mdl.process_image(image,
                      detector,
                      destiny_path,
                      image_names[i])

# Get faces information
faces_names, faces_files = get_images_info(f"{current_directory}/{faces_path}")

# Generate embeddings
embeddings = mdl.gen_embeddings(faces_files, processor, device, model)
# print(embeddings)

# Save embeddings to DB
for i in range(0, len(faces_files)):
    qd.insert_image_embedding(client, collection, i, image_names, embeddings)

img_to_search = mdl.gen_embedding_img_to_search(f"{current_directory}/Marcelo_Rebelo_de_Sousa_5.jpg", processor, device, model, detector)
# print(img_to_search)

# Search top 5 similar results
nearest = client.query_points(
    collection_name=collection,
    query=img_to_search[0],
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