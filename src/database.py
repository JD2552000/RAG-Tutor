#Now as we have chunks from the ingestion phase, we need a place to store them where they can be searched instantly.
#We will used Qdrant for this purpose mainly it saved the data to a folder vector_db/ so we dont need to re-process the data everytime we start the app.

import os
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

load_dotenv()

class VectorDatabase:
    def __init__(self, collection_name='tutor_knowledge'):
        self.collection_name = collection_name
        self.persist_path = 'vector_db'

        #Check if the directory exists
        if not os.path.exists(self.persist_path):
            os.makedirs(self.persist_path)
        
        #Initialize Google Embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

        #Initialize Qdrant Client
        self.client = QdrantClient(path=self.persist_path)

        #Create a Langchain wrapper for easy use
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection=self.collection_name,
            embedding=self.embeddings
        )

    def add_documents(self, chunks):
        #Uploads semantic chunks to our local Qdrant database
        if not chunks:
            return
        print(f'Indexing {len(chunks)} chunks into DB')
        self.vector_store.add_documents(chunks)
        print(f'Database updated successfully')

    def get_retriever(self, search_type='similarity', k=4):
        #Returns Retriever object to fetch the context
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={'k': k}
        )