#This file takes the context that is chunks from retreiver and then answers the questions.

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from src.retrieval import AdvancedRetriever
import os

class TutorEngine:
    def __init__(self):
        self.retriever = AdvancedRetriever()
        self.llm = ChatGoogleGenerativeAI(
            model='gemini-1.5-pro',
            temperature=0.3
        )

    def ask(self, question:str):
        #Fetch the context
        context_chunks = self.retreiver.search(question)
        context_text = '\n\n'.join(context_chunks)

        #The Main Prompt in which we will embed the question and the context
        template = """
        You are a professional Enterprise AI Tutor. Use the provided context to answer the student's question.
        
        Rules:
        1. If the answer isn't in the context, say "I don't have enough information in my database."
        2. Keep the explanation clear and structured.
        3. Use bullet points for complex topics.

        Context: {context}
        
        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm

        response = chain.invoke({'context': context_text, 'question': question})

        return response.content