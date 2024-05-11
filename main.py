# Integrate the openAi key with the application

import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

import streamlit as st

os.environ["OPENAI_API_KEY"] = openai_key

# streamlit Framework

st.title('Famous Person Search')
input_text = st.text_input('Search the topic you want to search')

#  Propmt Templates

first_input_prompt = PromptTemplate(
    # input_variables=["input_text"],
    # template=f"""
    # You are a famous person search engine.
    # You are given a topic and you have to search for the famous person related to that topic.
    # Search for the famous person related to {input_text}.
    # """

    input_variables=["name"],
    template= "Tell me about celebrity {name}"
)

# OPENAI LLMS
llm = OpenAI(temperature=0.8)
chain = LLMChain(llm = llm, prompt=first_input_prompt, verbose=True)

if input_text:
    st.write(chain.run(input_text))