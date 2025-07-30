from langchain.vectorstores import FAISS
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.chat_models import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 1. Load vectorstore
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
vectorstore = FAISS.load_local("vectorstore/fin_rag_index", embedding)

# 2. Load Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2, google_api_key=GOOGLE_API_KEY)

# 3. Build RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# 4. Ask a question
query = input("Ask your financial question: ")
response = qa_chain.run(query)
print("\n📊 Answer:\n", response)
