
# Standard library imports
import os
import io
import json

# Third-party library imports
import PyPDF2
import requests
import numpy as np
import cv2
import math
import pdfplumber
import matplotlib.pyplot as plt
from PIL import Image
import ipywidgets as widgets
from IPython.display import display, Markdown

# Google generative AI imports
import google.generativeai as genai  

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.document_loaders import TextLoader
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# Chromadb imports
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

# MediaPipe imports
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Import streamlit
import streamlit as st

google_api_key = os.environ["GEMINI_API_KEY"]
if google_api_key is None:
    st.warning("API key not found. Please set the google_api_key environment variable.")
    st.stop()

st.title("Customer Service Assistant") 
st.write("") 
st.write("") 
st.subheader("How can I help you today?")
st.write("") 

# Sidebar
st.sidebar.title("Customer Support")
st.write("") 
st.write("") 
st.sidebar.markdown("Use this assistant to get help with:")
st.sidebar.markdown("- Shoe recommendations")
st.sidebar.markdown("- General inquiries")

# Add the rest of the code here
# Footer
st.write("") 
st.write("") 
st.markdown("---")
st.markdown("### About Us")
st.markdown("We offer a wide range of shoes for all occasions. Feel free to browse our collection and reach out if you have any questions!")
st.markdown("Contact us at: [support@shoestore.com](mailto:support@shoestore.com)")

# Method to generate chunks
def get_text_chunks_langchain(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=1)
    chunks = text_splitter.split_text(text)
    docs = [x for x in chunks]
    return docs


def process_pdf(file_path):
    # PDF processing
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        pdf_pages = pdf_reader.pages

        # Create chunks for each shoe detail page
        shoe_details = []
        for page_num, page in enumerate(pdf_pages, start=1):

            page_text = page.extract_text()
            # Split the text into lines and remove any empty lines
            lines = [line.strip() for line in page_text.splitlines() if line.strip()]

            # Join lines into a continuous text
            consolidated_text = ' '.join(lines)
            shoe_details.append(consolidated_text)

            # PDF reading is done

        # Generate chunks for each shoe detail
        shoe_chunks = []
        for shoe_detail in shoe_details:
              chunks = get_text_chunks_langchain(shoe_detail)
              shoe_chunks.append(chunks)

        return shoe_chunks




# Method to extract images from the PDF
def extract_images_from_pdf(pdf_path):
    images = []
    print("Extracted images from the PDF")
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for img in page.images:
                try:
                    # Extract image coordinates
                    x0, top, x1, bottom = img['x0'], img['top'], img['x1'], img['bottom']
                    # Crop image from page
                    cropped_image = page.within_bbox((x0, top, x1, bottom)).to_image()
                    # Convert cropped image to bytes
                    image_stream = io.BytesIO()
                    cropped_image.save(image_stream, format='PNG')
                    pil_image = Image.open(image_stream).convert('RGB')  # Ensure RGB mode
                    
                    images.append(np.array(pil_image))
                except Exception as e:
                    print(f"Error processing image: {e}")

    return images


# Fetch the text and images from the PDF
pdf_file_path ='ShoesStore.pdf'
shoe_chunks=process_pdf(pdf_file_path)
images = extract_images_from_pdf(pdf_file_path)


# Method to create mp images
def create_mp_image_from_np_array(image_np):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)


# Method to generate image embeddings
def embed_images_with_mediapipe(images):
    base_options = python.BaseOptions(model_asset_path='embedder.tflite')
    l2_normalize = True
    quantize = True
    options = vision.ImageEmbedderOptions(
        base_options=base_options, l2_normalize=l2_normalize, quantize=quantize)

    with vision.ImageEmbedder.create_from_options(options) as embedder:
        mp_images = [create_mp_image_from_np_array(image_np) for image_np in images]

        # Embed each image
        embedding_results = []
        for image in mp_images:
            embedding_result = embedder.embed(image)
            embeddings = np.array(embedding_result.embeddings[0].embedding, dtype=np.uint8)
            embedding_results.append(embeddings)


        return embedding_results  # Return the list of embedding arrays


# Image embeddings
image_embeddings= embed_images_with_mediapipe(images)