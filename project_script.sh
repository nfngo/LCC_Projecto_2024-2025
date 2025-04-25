#!/bin/bash
option="$1"
case ${option} in
    "start_docker")
        docker run -p 6333:6333 \
            -v $(pwd)/qdrant_storage:/qdrant/storage \
            qdrant/qdrant
    ;;
    "clean_up")
        sudo chown -R "${USER:=$(/usr/bin/id -run)}" qdrant_storage/
        rm -rf faces
        rm -rf qdrant_storage
    ;;
    "start_app")
        python3 initial_setup.py && streamlit run search_image.py
    ;;
esac
