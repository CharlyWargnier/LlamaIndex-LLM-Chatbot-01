# Streamlit
import streamlit as st

# OpenAI - Library for interacting with OpenAI's services
import openai

# Llama Index
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI

# Streamlit Image Selection Component
from streamlit_image_select import image_select

# PIL - Python Imaging Library, for working with images
from PIL import Image

# Streamlit Pills Component
from streamlit_pills import pills

import ssl


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# nltk.download("punkt")

# import nltk
# nltk.download('punkt')

st.set_page_config(
    page_title="LlamaIndex Chatbot: Chat with Celebrity Wikis!", page_icon="🦙"
)

# Streamlit app layout
st.subheader("🦙 LlamaIndex Chatbot: Chat with Celebrity Wikis!")
st.subheader("")

data = "https://www.ycombinator.com/blog/content/images/2022/02/pg.jpg"
data2 = "https://static01.nyt.com/images/2021/06/13/books/review/Smith/merlin_126481298_d4afd655-6a72-4f41-b8c8-00f1633315fb-superJumbo.jpg"
data3 = "https://pbs.twimg.com/profile_images/1221837516816306177/_Ld4un5A_400x400.jpg"

api_key = st.sidebar.text_input("Enter your OPENAI API KEY", type="password")

if api_key:
    openai.api_key = api_key
else:
    st.sidebar.warning("👆  Please enter a valid OpenAI API key.")

img = image_select(
    label="Choose a tech personality",
    images=[data, data2, data3],
    captions=["Paul Graham", "Jeff Bezos", "Satya Nadella"],
)

if img == data:
    text = "Paul_Graham.txt"
    with open(text, "r") as file:
        text = file.read()
    st.info("📖 You selected the :red[**Paul Graham**] Wiki")

elif img == data2:
    text = "Jeff_Bezos.txt"
    with open(text, "r") as file:
        text = file.read()
    st.info("📖 You selected the :red[**Jeff Bezos**]  Wiki")

else:
    text = "Satya_Nadella.txt"
    with open(text, "r") as file:
        text = file.read()
    st.info("📖 You selected the :red[**Satya Nadella**]  Wiki")

with st.expander("Click to view the selected Wiki"):
    st.write(text)

selected = pills(
    "Prompt suggestions",
    [
        "In November 2022, who received Bezos' $100-million Courage and Civility Award for children's literacy work?",
        "In September 2022, where was Jeff Bezos ranked on the Forbes 400 list and what was his net worth?",
    ],
    ["🎈", "🎈"],
)

st.code(selected)

user_input = st.chat_input("Ask something about this Wiki:")

# Create the ServiceContext using the OpenAI llm
service_context = ServiceContext.from_defaults(llm=OpenAI())

# Convert the text into a suitable format for your index (e.g., a list of documents)
data = [Document(text=text)]

# Load data and build index
index = VectorStoreIndex.from_documents(data, service_context=service_context)

# Yi's suggestions on the new LlamaIndex 0.80
chat_engine = index.as_chat_engine(chat_mode="context", verbose=True, streaming=True)
chat_engine._context_template = (
    "Context information from the wiki is below."
    "\n--------------------\n"
    "{context_str}"
    "\n--------------------\n"
)

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    response = chat_engine.chat(user_input)
    with st.chat_message("assistant"):
        st.info(response)
