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

# facebook/dino-vitb16 size: 768
# Vit Base Patch16 224 In21k size: 768
# Create qdrant collection
qd.create_collection(client, collection, 768, models.Distance.COSINE)

data = []

current_directory = os.getcwd()
extensions = (".jpg", ".jpeg", ".webp")

def get_images_data(path):
    data = []
    for file in os.listdir(f"{path}"):
        if file.endswith(extensions):
            file_name = file.split(".")[0].lower().replace("_"," ")
            name = "".join(c for c in file_name if c.isalpha() or c == ' ')[:-1] 
            img_data = {
                "name": name,
                "file_name": file,
                "img_obj": Image.open(os.path.join(f"{path}",file))
            }
            data.append(img_data)
    
    return data


data = get_images_data(f"{current_directory}/assets")

# Initialize models 
device, processor, model, detector = mdl.init_models()

# Create directory if needed
faces_path = "faces"
if not os.path.exists(faces_path):
    os.makedirs(faces_path)

destiny_path = f"{current_directory}/{faces_path}/"
for i, image_data in enumerate(data):
    mdl.process_image(image_data['img_obj'], 
                      detector,
                      destiny_path,
                      image_data['file_name'])

faces_data = get_images_data(f"{current_directory}/{faces_path}")

# Generate embeddings
embeddings = mdl.gen_embeddings(faces_data, processor, device, model)

# Save embeddings to DB
for img_data in embeddings:
    qd.insert_image_embedding(client, collection, img_data)