import os
os.environ['MPLCONFIGDIR'] = '/tmp'


import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import tempfile
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
import matplotlib.pyplot as plt
@st.cache_resource
def load_captioning_model():
    return tf.keras.models.load_model("image_caption_model.keras")

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

model = load_captioning_model()
tokenizer = load_tokenizer()
max_length = 36  # Replace with actual value from training

# Load encoder model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

base_model = InceptionV3(weights="imagenet")
encoder_model = Model(base_model.input, base_model.layers[-2].output)
def preprocess_image(img_path):
    img = load_img(img_path, target_size=(299, 299))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def generate_caption(model, tokenizer, photo, max_length):
    in_text = '<start>'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')[0]
        sequence = np.array(sequence, dtype=np.int32).reshape(1, max_length)
        photo = np.reshape(photo, (1, 2048))

        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break

    return in_text.replace('<start>', '').replace('<end>', '').strip()

def caption_this_image(image_path, model, tokenizer, encoder_model, max_length):
    img = preprocess_image(image_path)
    feature = encoder_model.predict(img, verbose=0)
    caption = generate_caption(model, tokenizer, feature, max_length)
    return caption
def preprocess_image(img_path):
    img = load_img(img_path, target_size=(299, 299))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def generate_caption(model, tokenizer, photo, max_length):
    in_text = '<start>'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')[0]
        sequence = np.array(sequence, dtype=np.int32).reshape(1, max_length)
        photo = np.reshape(photo, (1, 2048))

        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break

    return in_text.replace('<start>', '').replace('<end>', '').strip()

def caption_this_image(image_path, model, tokenizer, encoder_model, max_length):
    img = preprocess_image(image_path)
    feature = encoder_model.predict(img, verbose=0)
    caption = generate_caption(model, tokenizer, feature, max_length)
    return caption

st.title("🖼️ Image Captioning App")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    temp_file.close()

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    st.write("Generating caption...")
    caption = caption_this_image(temp_file.name, model, tokenizer, encoder_model, max_length)

    st.success("Generated Caption:")
    st.markdown(f"**{caption}**")
