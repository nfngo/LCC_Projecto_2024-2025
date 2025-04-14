from qdrant_client import QdrantClient
from qdrant_client.http import models
from PIL import Image
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

# Initialize models 
device, processor, model, detector = mdl.init_models()

# Create directory if needed
faces_path = "faces"
if not os.path.exists(faces_path):
    os.makedirs(faces_path)

# Save faces as images in faces folder
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
    qd.insert_image_embedding(client, collection, i, faces_names, embeddings)