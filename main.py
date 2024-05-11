import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory

import streamlit as st

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment
openai_key = os.getenv("OPENAI_API_KEY")

# Set the API key in the environment
os.environ["OPENAI_API_KEY"] = openai_key

# streamlit Framework
st.title('Famous Person Search')
input_text = st.text_input('Search the topic you want to search')

# Prompt Templates
first_input_prompt = PromptTemplate(
    input_variables=["name"],
    template= "Tell me about celebrity {name}"
)

# Memory
person_memory = ConversationBufferMemory(imput_key = "name", memory_key='name_history')
dob_memory = ConversationBufferMemory(imput_key = "person", memory_key='dob_history')
events_memory = ConversationBufferMemory(imput_key = "dob", memory_key='events_history')

# OPENAI LLMS
llm = OpenAI(temperature=0.8)
chain1 = LLMChain(llm = llm, prompt=first_input_prompt, verbose=True, output_key='person', memory=person_memory)

second_input_prompt = PromptTemplate(
    input_variables=["person"],
    template= "When was {person} born"
)

chain2 = LLMChain(llm = llm, prompt=second_input_prompt, verbose=True, output_key='dob', memory=dob_memory)

third_input_prompt = PromptTemplate(
    input_variables=["dob"],
    template= "What 5 major events occurred on {dob}"
)

chain3 = LLMChain(llm = llm, prompt=third_input_prompt, verbose=True, output_key='events', memory=events_memory)

parent_chain = SequentialChain(chains=[chain1, chain2, chain3], input_variables=['name'], output_variables=['person', 'dob', 'events'], verbose=True)

# Modify the run method in SequentialChain
def run(self, inputs):
    outputs = {}
    for chain in self.chains:
        chain_outputs = chain.run(inputs)
        outputs.update(chain_outputs)
    return {key: outputs[key] for key in self.output_variables}

# Bind the run method to the class
SequentialChain.run = run

if input_text:
    result = parent_chain.run({'name': input_text})
    st.write(result)

    with st.expander('Person Name'):
        st.info(person_memory.buffer)

    with st.expander('Events Occurred'):
        st.info(events_memory.buffer)
