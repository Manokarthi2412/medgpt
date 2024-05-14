from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import constants
import pickle
import faiss
import os 

# Initialize Hugging Face embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Get the current directory path
current_dir = os.path.dirname(os.path.abspath(__file__))
# Paths for the FAISS store and the document index
faiss_store_path = os.path.join(current_dir, "faiss_store.pkl")
docs_index_path = os.path.join(current_dir, "docs.index")

# Load FAISS store and document index
with open(faiss_store_path, "rb") as f:
    store = pickle.load(f)
    store.index = faiss.read_index(docs_index_path)

# Initialize ChatGPT model
model = ChatOpenAI()

# Function to generate medical response using GPT model
def med_gpt(query):
    # Perform similarity search to find relevant documents
    relevant_docs = store.similarity_search(query, k=3)
    # Generate a prompt template based on the query
    prompt = PromptTemplate.from_template(query)
    # Create an LLMChain with the ChatGPT model and the prompt
    chain = LLMChain(llm=model, prompt=prompt)
    # Generate response from the chain, considering relevant documents as context
    response = chain.run(query=query, context=relevant_docs)
    return response


# Example usage:
# query = "High temperature with running nose"
# answer  = med_gpt(query)
# print(answer)
    


    
    
    
