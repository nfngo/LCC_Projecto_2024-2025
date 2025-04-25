from qdrant_client import QdrantClient
import os
import streamlit as st

import qdrant.utils as qd
import models.models as mdl

# Using containers
# client = QdrantClient(host="qdrant", port=6333)

# Local dev
client = QdrantClient(host="localhost", port=6333)

collection = "image_collection"

current_directory = os.getcwd()

# Initialize models
device, processor, model, detector = mdl.init_models()

st.title("Facial Recognition System")
st.markdown("Upload images with different faces and you'll get the most similar ones from our database.")

uploaded_file = st.file_uploader(label = "Upload some image",
                                 type=["jpg", "jpeg"])

# Search top X similar results
top = st.radio(
    "How many search results do you want?",
    [1, 3, 5],
)

with st.spinner("Waiting for results..."):
    if uploaded_file:
        img_to_search, searched_faces = mdl.gen_embedding_img_to_search(uploaded_file, processor, device, model, detector)


        nearest_results = []
        for img in img_to_search:
            nearest = qd.get_top_x_similar_images(client, collection, top, img)
            nearest_results.append(nearest)

        st.markdown("## Results:")

        def print_results(results):
            msg = f"{len(results)} faces were found in the provided image" if len(results) > 1 else f"{len(results)} face was found in the provided image"
            st.markdown(msg)

            for j, result in enumerate(results):
                points = result.points
                st.markdown(f"Face {j+1}:")
                st.image(searched_faces[j], width = 200)
                for i in range(len(points)):
                    name = points[i].payload['name']
                    score = points[i].score
                    st.markdown(f"Result #{i+1}: {name} was diagnosed with {score * 100} confidence")
                    st.markdown(f"This image score was {score}")
                    st.image(f"faces/{points[i].payload['file_name']}")
                    st.markdown("-" * 50)


        print_results(nearest_results)