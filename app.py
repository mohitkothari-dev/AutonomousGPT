import os
from dotenv import load_dotenv

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

load_dotenv()

s = os.getenv('OPENAI_API_KEY')

st.title('Indian Recipee Helper')
prompt = st.text_input('ASk GPT what you wanna cook')

# Giving app a persona
# 1. List recipees
title_template = PromptTemplate(
    input_variables=['topic'],
    template='Suggest me some healthy Indian food using {topic}'
)

# 2. Ingredients
ingredients_template = PromptTemplate(
    input_variables=['title'],
    template='Tell me all the ingredients required to cook {title}'
)
# 3. Process of Cooking
cooking_process_template = PromptTemplate(
    input_variables=['title'],
    template='Walk me through the entire process of cooking {title}'
)


# LLM
llm = OpenAI(temperature=0.7)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title')
ingredients_chain = LLMChain(llm=llm, prompt=ingredients_template, verbose=True, output_key='ingredients')
cooking_process_chain = LLMChain(llm=llm, prompt=cooking_process_template, verbose=True, output_key='cooking') 
# Update the output variables in SequentialChain
sequential_chain = SequentialChain(chains=[title_chain, ingredients_chain, cooking_process_chain],
                                   input_variables=['topic'],
                                   output_variables=['title', 'ingredients', 'cooking'], 
                                   verbose=True)

if prompt:
    response = sequential_chain({'topic': prompt})
    with st.expander("All recipees"):
        st.info(response['title'])
    with st.expander("Ingredients"):
        st.info(response['ingredients'])
    with st.expander("Method"):
        st.info(response['cooking']) 