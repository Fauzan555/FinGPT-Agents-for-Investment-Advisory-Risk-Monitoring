from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 1. Load documents
loader = DirectoryLoader('./data', glob="**/*.txt", loader_cls=TextLoader)
docs = loader.load()

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(docs)

# 3. Embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

# 4. Create vectorstore
vectorstore = FAISS.from_documents(documents, embeddings)

# 5. Save it locally
vectorstore.save_local("vectorstore/fin_rag_index")
print("✅ Vectorstore saved to ./vectorstore/fin_rag_index")
