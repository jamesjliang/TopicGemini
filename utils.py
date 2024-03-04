'''
Module Name: utils.py
Author: James Liang
Date: 03/03/2024

Helper functions that are used across multiple modules within the project
'''

from api_keys import gemini_key
import os
from langchain_google_genai import (ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory,)

# Load API Key.
os.environ["GOOGLE_API_KEY"] = gemini_key

##### 1. LOAD #####
def initialise_llm_model():
    '''
    Initalises the Google Gemini LLM model with the following settings:
        - model          : Gemini-pro
        - temperature    : 0.0 - to reduce variability in outputs as much as possible
        - safety_settings: Overrides the gemini models HarmBlockThreshold to block nothing.

    Returns: LLM model that takes in a prompt as query.
    '''
    llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature = 0.0,
    safety_settings = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE, 
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE, 
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE, 
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
},
        )
    return llm

def load_docs(file):
    '''
    Load a txt file, and converts it into a list of elements.

    Args:
        file             : Location of txt file. For example: "data/nyt.txt"
    Returns:
        docs             : List of elements, where each element is a New York Times extract.
    '''
    docs = []
    with open(file, 'r') as fin:
        for line in fin:
            docs.append(f"{line.strip()}")
    return docs

##### 5: Output the topics to a text file #####
def output_topics_to_txt(topic_struc):
    '''Output the Topic Hierarchy generated through clustering and LLM topic refinement, into a txt file'''
    
    output_taxo_file = "outputs/output_topics.txt"
    with open(output_taxo_file, 'w') as file:
        categories = topic_struc["ROOT"]

        for idx, category in enumerate(categories):
            file.write(f"Topic cluster {idx}: {category}\n")
            subtopics = topic_struc[category]

            for subtopic in subtopics:
                file.write(f"    - {subtopic}\n")
            file.write("\n")