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

img_to_search, searched_faces = mdl.gen_embedding_img_to_search(f"{current_directory}/Marcelo_Montenegro.jpg", processor, device, model, detector)

# Search top X similar results
top = 3
nearest_results = []
for img in img_to_search:
    nearest = qd.get_top_x_similar_images(client, collection, top, img)
    nearest_results.append(nearest)

# print(nearest_results)

def print_results(results):
    msg = f"{len(results)} faces were found in the provided image" if len(results) > 1 else f"{len(results)} face was found in the provided image"
    print(msg)
    for j, result in enumerate(results):
        # print(f"{j}: {result.points}")
        points = result.points
        print(f"Face {j+1}:")
        # searched_faces[j].show()
        for i in range(len(points)):
            name = points[i].payload['name']
            score = points[i].score
            print(f"Result #{i+1}: {name} was diagnosed with {score * 100} confidence")
            print(f"This image score was {score}")
            # Image.open(f"faces/{points[i].payload['image_url']}").show()
            print("-" * 50)
            print()

print_results(nearest_results)