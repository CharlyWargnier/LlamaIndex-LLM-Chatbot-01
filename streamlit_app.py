import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from PIL import Image
from streamlit_image_select import image_select
import time
from streamlit_pills import pills

import nltk
nltk.download('punkt')

# Streamlit app layout
st.header("🦙 LlamaIndex Chatbot with Celebrity Wikis")

data = "https://www.ycombinator.com/blog/content/images/2022/02/pg.jpg"
data2 = "https://static01.nyt.com/images/2021/06/13/books/review/Smith/merlin_126481298_d4afd655-6a72-4f41-b8c8-00f1633315fb-superJumbo.jpg"
data3 = "https://helios-i.mashable.com/imagery/articles/05SzGlYqpD4cUlIaQ8DHdVF/hero-image.fill.size_1200x1200.v1666999270.jpg"

img = image_select(
    label="Choose a tech personality",
    images=[data, data2, data3],
    captions=["Paul Graham", "Jeff Bezos", "Elon Musk"],
)

if img == data:
    text = 'Paul_Graham.txt'
    with open(text, 'r') as file:
        text = file.read()
    st.info("📖 You selected the :red[**Paul Graham**] Wiki")

elif img == data2:
    text = 'Jeff_Bezos.txt'
    with open(text, 'r') as file:
        text = file.read()
    st.info("📖 You selected the :red[**Jeff Bezos**]  Wiki")

else:
    text = 'Elon_Musk.txt'
    with open(text, 'r') as file:
        text = file.read()
    st.info("📖 You selected the :red[**Elon Musk**]  Wiki")

selected = pills("Prompt ideas", ["What's this Wiki about?", "What's the most interesting fact about this Wiki?", "What controversies has this person faced in his life?"], ["🎈", "🎈", "🎈"])

# st.caption('Copy prompt')
st.code(selected)

user_input = st.chat_input("Ask something about this Wiki:")

# Check if user input is less than 100 words
if user_input and len(user_input.split()) < 3:
    st.error("Your message is too short. Please enter at least 3 words.")
else:
    # Set the OpenAI API key
    openai.api_key = "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"

    #os.environ['OPENAI_API_KEY'] = st.secrets['openai']['OPENAI_API_KEY']
    #openai.api_key = os.environ['OPENAI_API_KEY']
    
    # Create the ServiceContext using the OpenAI llm
    service_context = ServiceContext.from_defaults(llm=OpenAI())

    # Convert the text into a suitable format for your index (e.g., a list of documents)
    data = [Document(text=text)]

    # Load data and build index
    index = VectorStoreIndex.from_documents(data, service_context=service_context)

    # Configure chat engine
    chat_engine = index.as_chat_engine(chat_mode="react", verbose=True, streaming=True)

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        # stream = chat_engine.astream_chat(user_input)
        response = chat_engine.chat(user_input)
        with st.chat_message("assistant"):
            st.info(response)
