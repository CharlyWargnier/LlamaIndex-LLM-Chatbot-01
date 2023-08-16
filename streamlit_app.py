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
import os

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# nltk.download("punkt")
# import nltk
# nltk.download('punkt')

try:
    st.set_page_config(
        page_title="LlamaIndex Chatbot: Chat with Tech Personalities", page_icon="🦙"
    )

    cols = st.columns([0.4, 3]) # The left column will be 1/4 of the total width, and the right column will be 3/4
    cols[0].image('https://aeiljuispo.cloudimg.io/v7/https://cdn-uploads.huggingface.co/production/uploads/6424f01ea4f3051f54dbbd85/oqVQ04b5KiGt5WOWJmYt8.png?w=200&h=200&f=face', width=70)
    #cols[1].caption("")
    cols[1].header("Chat with Wikipedia using LLamaIndex")

    st.write('''

    This demo app uses [LlamaIndex](https://www.llamaindex.ai/) to build an LLM chatbot that can chat with your data. This contrasts with ChatGPT, which has been trained on data [only until 2021](https://bit.ly/3qwUqeQ). With LlamaIndex, you can train your own LLM on your own data, so that it can answer more specific and relevant questions.
    ''')

    
    Sergey_Brin_image = "https://image.cnbcfm.com/api/v1/image/102730650-152766135.jpg?v=1522952646"
    Jeff_Bezos_image = "https://fr.web.img6.acsta.net/pictures/22/08/31/17/40/2573138.jpg"
    Satya_Nadella_image = "https://content.fortune.com/wp-content/uploads/2022/02/Satya-Nadella-Microsoft-CEO-Most-Admired.jpg"

    # Read API key from Streamlit secrets
    #if 'OPENAI_API_KEY' in os.environ:
    openai.api_key = os.environ['OPENAI_API_KEY']
    #else:
        #st.sidebar.warning("👆  Please set the OpenAI API key in your Streamlit secrets.")
        # pass

    img = image_select(
        label="① Choose a tech personality",
        images=[Sergey_Brin_image, Jeff_Bezos_image, Satya_Nadella_image],
        captions=["Sergey Brin", "Jeff Bezos", "Satya Nadella"]
    )



    if img == Sergey_Brin_image:
        with open("Sergey_Brin.txt", "r") as file:
            text = file.read()
        st.info(f"📖 You selected the **Sergey Brin** Wiki. We scraped this data from his [Wikipedia page](https://en.wikipedia.org/wiki/Sergey_Brin).")

    elif img == Jeff_Bezos_image:
        with open("Jeff_Bezos.txt", "r") as file:
            text = file.read()
        st.info(f"📖 You selected the **Jeff Bezos** Wiki. We scraped this data from his [Wikipedia page](https://en.wikipedia.org/wiki/Jeff_Bezos).")

    else:
        with open("Satya_Nadella.txt", "r") as file:
            text = file.read()
        st.info(f"📖 You selected the **Satya Nadella** Wiki. We scraped this data from his [Wikipedia page](https://en.wikipedia.org/wiki/Satya_Nadella).")


    st.write("")

    selected = pills(
        "② Select a prompt",
        [
            "What's Brin net worth as of July 2023?",
            "What happened to Jeff Bezos in September 2022?",
            "What recognition did Nadella receive from the Government of India in 2022",
        ],
        ["🎈", "🎈", "🎈"],
    )


    st.code(selected, language="None")

    st.caption('③ Paste that prompt in the chat box')
    col1, col2, col3 = st.columns([0.11, 1, 1])
    with col1:
        arrow = "Images/blue_arrow.jpg"
        st.image(arrow, width=90)
        
    user_input = st.chat_input("Ask something about this Wikipedia page:")

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

except ValueError as e:
    st.error(f"An error occurred: {str(e)}")
