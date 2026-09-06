# Standard library imports
import os
import io
import getpass
import json

# Third-party library imports
import PyPDF2
import requests
import numpy as np
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

# Method to process the PDF
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

            # Join lines into continuous text
            consolidated_text = ' '.join(lines)
            shoe_details.append(consolidated_text)

            # PDF reading is done

        # Generate chunks for each shoe detail
        shoe_chunks = []
        for shoe_detail in shoe_details:
              chunks = get_text_chunks_langchain(shoe_detail)
              shoe_chunks.append(chunks)

        return shoe_chunks