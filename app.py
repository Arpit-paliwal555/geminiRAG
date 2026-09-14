
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

