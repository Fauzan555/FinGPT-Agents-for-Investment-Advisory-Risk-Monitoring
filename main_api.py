from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain.vectorstores import FAISS
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.chat_models import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# Load environment
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize app
app = FastAPI(title="FinGPT RAG API")

# Pydantic model for request
class QueryRequest(BaseModel):
    query: str

# Load vector store and LLM once
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
vectorstore = FAISS.load_local("vectorstore/fin_rag_index", embedding)

llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2, google_api_key=GOOGLE_API_KEY)
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

@app.post("/query")
async def query_financial_agent(request: QueryRequest):
    user_query = request.query
    result = qa_chain.run(user_query)
    return {"response": result}
