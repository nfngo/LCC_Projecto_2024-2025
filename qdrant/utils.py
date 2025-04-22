from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
import uuid

# Create collection
def create_collection(client: QdrantClient, 
                      collection_name: str, 
                      vector_size: int,
                      distance_measure: models.Distance = models.Distance.COSINE):
    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name = collection_name,
            vectors_config=  models.VectorParams(
                size = vector_size, 
                distance = distance_measure)
        )


# Insert image embedding
def insert_image_embedding(client: QdrantClient, 
                           collection_name: str, 
                           img_data: dict
                           ):
    
    operation_info = client.upsert(
        collection_name = collection_name,
        points = [models.PointStruct(
            id = str(uuid.uuid4()),
            payload = {
                "name": img_data['name'],
                "file_name": img_data['file_name']
            },
            vector = img_data['img_embedding'],
        )]
    )

    if operation_info.status == models.UpdateStatus.COMPLETED:
        print("Data inserted successfully")
    else:
        print("Failed to insert data")


# Retrieves {limit} most similar images to query embedding
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


# Get {limit} stored points 
def get_points_from_collection(client: QdrantClient, 
                           collection_name: str,
                           limit: int,
                           with_vectors: bool = False):
    return client.scroll(
        collection_name = collection_name,
        limit = limit,
        with_payload = True,
        with_vectors = with_vectors
    )


# Deletes specified points (by id) from the collection.
def delete_points_from_collection(client: QdrantClient, 
                           collection_name: str,
                           ids: list):
    operation_info = client.delete(
        collection_name = collection_name,
        points_selector = models.PointIdsList(
            points = ids
        )
    )
    
    if operation_info.status == models.UpdateStatus.COMPLETED:
        print("Points deleted successfully")
    else:
        print("Failed to delete points")