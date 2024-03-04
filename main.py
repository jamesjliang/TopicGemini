'''
Project Name: TopicGemini

Module Name: main.py
Author: James Liang
Date: 03/03/2024

Input : Text Corpus                         "nyt.txt"
Output: Topic Hierarchy of Corpus.          "outputs/output_topics.txt"
        - with RAG supplemented summarisations for finer granularity topic information.

Project Background:
Topic Modelling is a common, yet deeply fascinating component of Natural Language Processing.
Usually, traditional approaches to this problem involve either:
    (1) Latent Dirichlet Allocation (LDA) 
    (2) Clustering and extracting centroid terms.

However, both these methods suffer from unsatisfactory topic granularity in the extracted terms,
and often, require a professional human curator to relabel the extracted 'topic' terms.

With the introduction of Large Language Models such as GPT-4 or Gemini, which are capable of incorporating semantic understanding
into natural language processing, TopicGemi explores the potential of how LLMs can be leveraged to 
improve existing processes in Topic Modelling, making them more effective for topic extraction.

'''
import nltk
from gensim.models import Word2Vec
from utils import load_docs, output_topics_to_txt, initialise_llm_model
from clustering import (create_word2vec_model, process_words, run_clustering)
from LLM_topic_refine import refine_topic_clusters

from RAG_summarise import setup_vectorstore, get_topic_summary

##### 3 + 4: Recursively cluster (3) and Refine topic clusters (4) with LLM #####
def format_into_topic_structure(topic_structure, node_name, refined_topic_term_clusters):
    '''
    Assign child topics to parent node.

    Args:
        topic_structure             : Dictionary - {parent_node: [child_topics]}
        node_name                   : Parent node name
        refined_topic_term_clusters : List of tuples - [(topic, term_clusters)]
                                      Where topic refers to topic name. Term_clusters are a list of associated cluster terms.
    Returns:
        topic_structure             : Updated topic structure.
    '''
    topic_list = [topic for topic, term_cluster in refined_topic_term_clusters]
    topic_structure[node_name] = topic_list
    return topic_structure

def recusively_cluster(word2vec_model, level, semantic_words = None, semantic_vectors = None, node_name = None, k_clusters = 10, is_child = False):
    '''
    Recursively performs kmeans clustering and LLM topic refinement on all nodes of the topic hierarchy.

    Args:
        word2vec_model   : The word2vec model trained on the entire text corpus.
        level            : Depth level of current topic in the topic hierarchy.
        semantic_words   : List of associated words in the cluster (that have an embedding value)
        semantic_vectors : List of embedding vectors for each word
        node_name        : Parent node name
        k_clusters       : Initialises the number of clusters to create in kmeans clustering.
        is_child         : Defines whether current node is ROOT or not. Influences which prompts are used in LLMs.

    Returns: None. (builds up the TOPIC_STRUCTURE through running)
    
    '''
    if level == MAX_DEPTH: # If level reaches 2, then stop recursion
        if node_name not in TOPIC_STRUCTURE:
            TOPIC_STRUCTURE[node_name] = semantic_words                  # Place all cluster words into the leaf node
        return 
    if node_name == None:
        node_name = "ROOT"
        # 0. Get the word vectors and corresponding words of ALL words
        word_vectors = word2vec_model.wv
        words_with_embedding = list(word_vectors.key_to_index.keys()) 
        
        semantic_words = process_words(words_with_embedding)
        semantic_vectors = [word_vectors[word] for word in semantic_words]
        
    print('============================= Running depth ', level, ' on node ', node_name, '=============================')

    # a. Apply kmeans clustering on terms.
    print("Run k-means clustering...")
    topic_term_clusters = run_clustering(word2vec_model, semantic_words, semantic_vectors, k_clusters, node_name, is_child)
    print("    Done.\n")

    # b. Refine topics through combining terms.
    print("Using LLM to refine topic clusters...")
    refined_topic_term_clusters = refine_topic_clusters(topic_term_clusters, node_name, is_child)
    print("    Done.\n")

    # bi. Format topics into topic structure
    format_into_topic_structure(TOPIC_STRUCTURE, node_name, refined_topic_term_clusters)

    # c. Get all generated topic names
    for child_topic, term_cluster in refined_topic_term_clusters:
        # 0. Get the word vectors and corresponding words of CLUSTER SPECIFIC words
        word_vectors = word2vec_model.wv
        semantic_words = term_cluster
        semantic_vectors = [word_vectors[word] for word in semantic_words]

        # 0.1. Set is_child to True - This alters the LLM prompts going forward.
        recusively_cluster(word2vec_model, level+1, semantic_words, semantic_vectors, node_name = child_topic, k_clusters = 4, is_child = True)

def main(input_corpus):
    '''
    Using only the input_corpus, create the Topic Hierarchy of the Corpus, 
    alongside its most prevalent, high granularity subtopics.

    Args:
        input_corpus: "data/nyt.txt"

    Outpus txt file can be found at: 
        "outputs/output_topics.txt" 

    Methodology:
    1. Input news corpus
    2. Word2Vec for embeddings.
    Recursively:
        3. Kmeans for clustering.
        4. LLMs (Gemini) for cluster refinement - using centroid similar terms.
    5. Output to txt file.
    '''
    # Step 1: Load in the input corpus
    print("1. Loading txt file")
    corpus = load_docs(input_corpus)
    print("    Done.\n")

    # Step 2: Train embedding model on input corpus
    print("2. Train word2vec embedding model")
    if not EMBEDDING_MODEL_CREATED:
        print("    Creating the word2vec embedding model...")
        word2vec_model = create_word2vec_model(corpus, save_model=True)
    word2vec_model = Word2Vec.load('models/word2vec_model.bin')
    print("    Done.\n")

    # Step 3 + 4: Recursively cluster (3) and Refine topic clusters (4) with LLM
    recusively_cluster(word2vec_model, 0)

    # Step 5: Output the topics to a text file
    output_topics_to_txt(TOPIC_STRUCTURE)

def demonstrate_RAG():
    '''
    A demonstration for how users can interact with the generated topic hierarchy - using Retrieval Augemented Generation (RAG).
    
    I allow users to query about any particular topic generated in main().
        - RAG allows the retrieval of topic relevant documents
        - A summarisation of these relevant documents...

    ...to output both the (1) summarisation of the topic with context, and (2) the index of documents retrieved.
    '''
    # Step 6: Output an example of RAG extraction of a topic.
    vectorstore = setup_vectorstore(VECTOR_STORE_CREATED)        # If VECTOR_STORE_CREATED is False: Vector Store will be created and saved to local disc following first run.

    #### Example of RAG.
    prompt = "Summarise the news extracts about the sports topic, champions league finals"
    print(f"\n\nRunning RAG on prompt: {prompt}\n")
    ai_output = get_topic_summary(vectorstore, prompt)

    # Step 6.1: RAG to extract specific NYT articles regarding topics discovered. (Write this into a file.)
    print(f"    Summary: {ai_output['answer']}\n")
    print(f"    Documents used: {ai_output['documents']}")

def test_valid_api_key():
    '''Test initialising an LLM model with the API key'''
    llm = initialise_llm_model()

if __name__ == "__main__":
    STOPWORDS = set(nltk.corpus.stopwords.words('english'))
    MAX_DEPTH = 2
    TOPIC_STRUCTURE = {}
    EMBEDDING_MODEL_CREATED = False            # Create a new Word2Vec Model that is saved to local disc (~1 min). Set to True after first run. 
    VECTOR_STORE_CREATED = False               # Create a new Vector Store that will be saved to local disc (~4 mins). Set to True after first run.
    test_valid_api_key()

    # Topic Modelling on input corpus
    input_corpus = "data/nyt.txt"
    main(input_corpus)

    # RAG Demonstration on topic. [Unhash demonstrate_RAG() to create a vector store and run a demonstration prompt with RAG)
    # demonstrate_RAG()
