import streamlit as st
from langchain_community.document_loaders import NotionDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings

from pymongo import MongoClient
from langchain_community.vectorstores import MongoDBAtlasVectorSearch

# Load the Notion content located in the folder 'notion_content'
loader = NotionDirectoryLoader("notion_content")
documents = loader.load()

# Split the Notion content into smaller chunks
markdown_splitter = RecursiveCharacterTextSplitter(
    separators=["#","##", "###", "\\n\\n","\\n","."],
    chunk_size=1500,
    chunk_overlap=100)
docs = markdown_splitter.split_documents(documents)

# Initialize OpenAI embedding model
openai_api_key = st.secrets['OPENAI_API_KEY']
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-ada-002")

# Convert all chunks into vectors embeddings and store embedding in MongoDB
mongodb_uri = st.secrets['MONGODB_ATLAS_CLUSTER_URI']
cluster = MongoClient(mongodb_uri)
DB_NAME = "notion-demo"
COLLECTION_NAME = "employee_handbook"
VECTOR_SEARCH_INDEX_NAME = "default"
collection = cluster[DB_NAME][COLLECTION_NAME]

# make sure to whitelist IP in mongoDB -> Security -> Network Access
# reference https://python.langchain.com/docs/integrations/vectorstores/mongodb_atlas
vector_search = MongoDBAtlasVectorSearch.from_documents(
    documents=docs,
    embedding=embeddings,
    collection=collection,
    index_name=VECTOR_SEARCH_INDEX_NAME
)
