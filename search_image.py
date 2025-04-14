from qdrant_client import QdrantClient
import os

import qdrant.utils as qd
import models.models as mdl

# Using containers
# client = QdrantClient(host="qdrant", port=6333)

# Local dev
client = QdrantClient(host="localhost", port=6333)

collection = "image_collection"

current_directory = os.getcwd()

device, processor, model, detector = mdl.init_models()

img_to_search = mdl.gen_embedding_img_to_search(f"{current_directory}/Antonio_Marcelo.jpg", processor, device, model, detector)

# Search top 5 similar results
nearest_results = []
for img in img_to_search:
    nearest = qd.get_top_x_similar_images(client, collection, 3, img)
    nearest_results.append(nearest)

# print(nearest_results)

def print_results(results):
    print(f"{len(results)} faces were found in the provided image")
    for j, result in enumerate(results):
        # print(f"{j}: {result.points}")
        points = result.points
        print(f"Face {j+1}:")
        for i in range(len(points)):
            name = points[i].payload['name']
            score = points[i].score
            print(f"Result #{i+1}: {name} was diagnosed with {score * 100} confidence")
            print(f"This image score was {score}")
            print("-" * 50)
            print()

print_results(nearest_results)