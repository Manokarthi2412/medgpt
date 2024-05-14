import pickle
import faiss
import langchain
import pypdf
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings

pdf_directory = "\path\directory"
# pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
pages= []
#embedding model declaration 
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

pdf_files = ["\documents\doc_1.pdf","\documents\doc_2.pdf","\documents\doc_3.pdf","\documents\doc_4.pdf","\documents\doc_5.pdf"]
for files in pdf_files:
    print(type(files))
    path = pdf_directory+files
    loader = PyPDFLoader(path)
    pages += loader.load_and_split()
    print(type(pages))

#Creating the vectorstore 
faiss_index = FAISS.from_documents(pages,embeddings)

faiss.write_index(faiss_index.index, "docs.index")
with open("faiss_store.pkl", "wb") as f:
    pickle.dump(faiss_index, f)


