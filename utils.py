import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain_community.vectorstores import MongoDBAtlasVectorSearch


@st.cache_resource
def create_vector_search(_collection):
    """
    Initialize vector search
    :param _collection: vector database collection
    :return:
    """
    # Load OpenAI embedding model
    embeddings = OpenAIEmbeddings()

    # Create vector store
    vector_store = MongoDBAtlasVectorSearch(_collection, embeddings)

    return vector_store


@st.cache_resource
def load_chain(_vector_store):
    """
    The `load_chain()` function initializes and configures a conversational retrieval chain for
    answering user questions.
    :param _vector_store: vector store that can perform vector search
    :return: The `load_chain()` function returns a ConversationalRetrievalChain object.
    """
    # Load OpenAI chat model
    llm = ChatOpenAI(temperature=0)

    # Load vector store as retriever
    retriever = _vector_store.as_retriever(search_kwargs={"k": 3})

    # Create memory 'chat_history'
    memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history")

    # Create system prompt
    template = """
    You are an AI assistant for answering questions about the Blendle Employee Handbook.
    You are given the following extracted parts of a long document and a question. Provide a conversational answer.
    If you don't know the answer, just say 'Sorry, I don't know ...'. 
    Don't try to make up an answer.
    If the question is not about the Blendle Employee Handbook, politely inform them that you are tuned to only answer questions about the Blendle Employee Handbook.

    {context}
    Question: {question}
    Helpful Answer:"""

    # Create the Conversational Chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        get_chat_history=lambda h: h,
        verbose=True
    )

    # Add systemp prompt to chain
    # Can only add it at the end for ConversationalRetrievalChain
    chain_prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    chain.combine_docs_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate(prompt=chain_prompt)

    return chain
