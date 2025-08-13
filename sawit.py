import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image


st.set_page_config(page_title="Halaman Modelling", layout="wide")
st.write("""
# Welcome to my portofolio Data Analyst
Hello my name is [Bramantio](https://www.linkedin.com/in/brahmantio-w/) and I am passionate about uncovering insights through data. 
\n With a strong enthusiasm for data analysis, I enjoy transforming raw information into meaningful stories that drive better decision-making.

""")
st.header("Implementasi Convolutional Neural Network untuk Identifikasi Tingkat Kematangan Buah Kelapa Sawit")
st.write("Dataset yang digunakan berasal dari [kaggle](https://www.kaggle.com/datasets/ramadanizikri112/ripeness-of-oil-palm-fruit)")
st.write("Submit gambar kelapa sawit yang anda miliki, kemudian model akan mengklasifikasikan gambar antara Belum Matang, Matang, atau Terlalu Matang")

 # Load model
model = load_model("sawit_model.keras",compile=False)

# Upload file
uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
            # tampilkan
            img_pil = Image.open(uploaded_file).convert("RGB")
            st.image(img_pil, caption="Gambar yang diunggah", use_column_width=True)

            # resize & preprocess
            img_resized = img_pil.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)


            # Prediksi
            preds = model.predict(img_array)
            predicted_class = np.argmax(preds, axis=1)
    
            # Mapping kelas ke label
            label_mapping = {
                0: "Belum Matang",
                1: "Matang",
                2: "Terlalu Matang"
                }
            label_prediksi = label_mapping[predicted_class[0]]

            st.write(f"Hasil Prediksi: {label_prediksi}")
