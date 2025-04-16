from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np

# Create collection
def create_collection(client: QdrantClient, 
                      collection_name: str, 
                      vector_size: int,
                      distance_measure: models.Distance = models.Distance.DOT):
    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name = collection_name,
            vectors_config=  models.VectorParams(
                size = vector_size, 
                distance = distance_measure)
        )


# Insert/update image embedding in the specified qdrant collection
def insert_image_embedding(client: QdrantClient, 
                           collection_name: str, 
                           img_id: int, 
                           img_names: list[str],
                           img_embeddings: np.ndarray
                           ):
    file_name = img_names[img_id].lower().replace("_"," ")
    name = "".join(c for c in file_name if c.isalpha() or c == ' ')
    client.upsert(
        collection_name = collection_name,
        points = [models.PointStruct(
            id = img_id,
            payload = {
                "name": name,
                "image_url": f"{img_names[img_id]}.jpg"
            },
            vector = img_embeddings[img_id][0],
        )]
    )


# Retrives x most similar images to query embedding
def get_top_x_similar_images(client: QdrantClient, 
                           collection_name: str,
                           limit: int,
                           query_embedding: list[float]):
    return client.query_points(
        collection_name = collection_name,
        query = query_embedding,
        limit = limit,
        with_payload = True
        )