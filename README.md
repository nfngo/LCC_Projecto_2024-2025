# LCC_Projecto_2024-2025

- Start Qdrant

docker run -p 6333:6333 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant