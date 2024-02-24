
import time
import streamlit as st
from pymongo import MongoClient
from utils import load_chain, create_vector_search

# Custom image for the app icon and the assistant's avatar
logo = 'shiba.png'

# Configure Streamlit page
st.set_page_config(
    page_title="Your Notion Chatbot",
    page_icon=logo
)

# Connect to MongoDB
mongodb_uri = st.secrets['MONGODB_ATLAS_CLUSTER_URI']
cluster = MongoClient(mongodb_uri)
DB_NAME = "notion-demo"
COLLECTION_NAME = "employee_handbook"
VECTOR_SEARCH_INDEX_NAME = "default"
collection = cluster[DB_NAME][COLLECTION_NAME]
vector_store = create_vector_search(collection)

# Initialize LLM chain
chain = load_chain(vector_store)

# Initialize chat history
if 'messages' not in st.session_state:
    # Start with first message from assistant
    st.session_state['messages'] = [{"role": "assistant",
                                     "content": "Hi, I am Shiba AI. How can I help you today?"}]

# Display chat messages from history on app rerun
# Custom avatar for the assistant, default avatar for user
for message in st.session_state.messages:
    if message["role"] == 'assistant':
        with st.chat_message(message["role"], avatar=logo):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat logic
if query := st.chat_input("Ask me anything"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant", avatar=logo):
        message_placeholder = st.empty()
        # Send user's question to our chain
        result = chain({"question": query})
        response = result['answer']
        full_response = ""

        # Simulate stream of response with milliseconds delay
        for chunk in response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})