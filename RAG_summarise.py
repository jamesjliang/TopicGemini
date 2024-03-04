'''
Module Name: RAG_summarise.py
Author: James Liang
Date: 03/03/2024

Use Retrieval Augmented Generation (RAG) to provide even finer
granularity summary of a specific topic that is being discussed.
'''

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader 
from langchain.document_loaders import TextLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma                               
from langchain.prompts import PromptTemplate
# For using retriever without metadata
from langchain.schema import StrOutputParser
# Run Parallel
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel

from utils import initialise_llm_model
import pandas as pd

def create_vectorstore(file, for_document = True):
    '''
    Create a vector store by through the processing of:
    1. Loading a Document object
    2. Chunking the document (if: for_document = False)
    3. Embedding via GoogleGenerativeAIEmbeddings, and using Chroma for VectorStore.

    Note that chunking the documents and creating a VectorStore that way, has a significant runtime as compared to embedding individual documents.
    '''
    # Create Gemini embedding model
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if for_document:
        # Create Vector Store by embedding each individual document.
        persist_directory = "models/doc_vectorstore"

        # 0. Convert arxiv.txt -> arxiv.csv
        abstracts = []
        with open(file, 'r') as fin:
            for line in fin:
                abstracts.append(line.strip())
        df = pd.DataFrame({"documents":[abstract for abstract in abstracts]})
        df.to_csv('outputs/nyt.csv', index = False)

        # 1. Load Documents Object
        loader = CSVLoader("outputs/nyt.csv")
        documents = loader.load()

        # 2. Embed and store in Vector Store (JSON docs more efficient)
        vectorstore = Chroma.from_documents(documents, embedding_model, persist_directory=persist_directory)

    else:
        # Create Vector Store by embedding each CHUNK of the document.
        persist_directory = "models/chunk_vectorstore"

        # 1. Load Documents Object
        raw_documents = TextLoader(file).load()

        # 2. Chunk Document
        text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
        )
        documents = text_splitter.split_documents(raw_documents)
    
        # 3. Embed and store in Vector Store (JSON docs more efficient)
        vectorstore = Chroma.from_documents(documents, embedding_model, persist_directory=persist_directory)

    return vectorstore

def format_docs(docs, flag = True):
    '''Format the documents'''
    if flag:
        return "\n\n".join(doc.page_content for doc in docs)
    else:
        result = ""
        for doc in docs:
            result += f"Doc id: {doc.metadata['row']}  "        # Doc id is Represented by metadata (row number).
            result += str(doc.page_content) + "\n\n"
        return result
    
def get_topic_summary(vectorstore, prompt, for_document = True):
    '''
    Implementation of RAG for extracting finer granularity summary of a specific topic generated from TopicGemi.

    Args:
        vectorstore : Chroma vectorstore.
        prompt      : Query in the format - "Summarise the news extracts about {topic}"
        for_document: Flag that determines the type of retriever to create for RAG.
    
    Returns:
        ai_output   : A dictionary which contains - (1) The summary response to the prompt. (2) The news extracts used in its response relevant to the topic.
    '''

    llm = initialise_llm_model()

    # 1. Setup Retriever which gets the most relevant documents to a prompt
    if for_document:
        # Documents. Use less examples.
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 15})
    else:
        # Chunks. Use more examples.
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 50})

    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Context: {context}

    Question: {question}
    Helpful Answer:"""
    rag_prompt_custom = PromptTemplate.from_template(template)

    #####################

    rag_chain_from_docs = (
        {
            "context": lambda input: format_docs(input["documents"]),
            "question": itemgetter("question"),
        }
        | rag_prompt_custom
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"documents": retriever, "question": RunnablePassthrough()}
    ) | {
        "documents": lambda input: [doc.metadata for doc in input["documents"]],
        "answer": rag_chain_from_docs,
    }

    ai_output = rag_chain_with_source.invoke(prompt)

    return ai_output

def setup_vectorstore(vectorstore_created = True, for_doc = True):  
    '''
    Setup for vector store.

    Args:
        vectorstore_created: Flag indicating whether a vector store has been created or not.
        for_doc            : Flag indicating whether to chunk documents, or just embed for each individual doc.
    
    Returns:
        vectorstore        : Chroma vector store with GoogleGenerativeAIEmbeddings
    '''
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if for_doc:
        persist_directory = "models/doc_vectorstore"
    else:
        persist_directory = "models/chunk_vectorstore"

    # Step 6: Create vector store for Retrieval Augemented Generation (RAG) to generate Summaries of topics discovered
    if vectorstore_created:
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function = embedding_model)
    else:
        vectorstore = create_vectorstore(file, for_document = for_doc)

    return vectorstore

################################################################################
if __name__ == "__main__":
    # Step 0: Setup
    file = "data/nyt.txt"
    vectorstore_created = True  # If True, we will NOT create a Vector Store. We will load an existing one.
    for_doc = True              # Create a Vector Store embedding each document. Set to False, if embedding as chunks.
    vectorstore = setup_vectorstore(vectorstore_created, for_doc)

    #### Example
    prompt = "Summarise the news extracts about the sports topic, champions league finals"
    ai_output = get_topic_summary(vectorstore, prompt, for_document = for_doc)

    # Step 6.1: RAG to extract specific NYT articles regarding topics discovered.
    print(f"Summary: {ai_output['answer']}")
    print(f"Documents used: {ai_output['documents']}")