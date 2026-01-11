#This file is for processing the data and converting it into a format that can be used by the model basically vectorizing the data.
#We will use Semantic Chunker so that the data is chunked based on the context/meaning of the data and not just the character count.
#Also Google's Embeddings is used to power the Semantic Chunker instead of Open Source Embeddings.

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

class DataIngestor:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")    

        #This is advanced splitter that group sentences by meaning similarity.
        self.splitter = SemanticChunker(self.embeddings)

    def load_and_split(self, data_path: str):
        print(f'Loading documents from {data_path}')
        #Load pdf files
        pdf_loader = DirectoryLoader(data_path, glob='**/*.pdf', loader_cls=PyPDFLoader)
        #Load text files
        txt_loader = DirectoryLoader(data_path, glob='**/*.txt', loader_cls=TextLoader)

        docs = pdf_loader.load() + txt_loader.load()

        if not docs:
            raise ValueError(f'No documents found in {data_path}')
            return []
        print(f'Splitting {len(docs)} documents into semantic chunks')
        chunks = self.splitter.split_documents(docs)

        print(f'Created {len(chunks)} semantic chunks')
        return chunks

if __name__ == '__main__':
    #Test script for ingestion phase
    ingestor = DataIngestor()
    # chunks = ingestor.load_and_split('data/')