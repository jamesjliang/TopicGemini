'''
Module Name: clustering.py
Author: James Liang
Date: 03/03/2024

Creating the word embeddings (word2vec) and applying k-means clustering.
Topic extraction is handled by the LLM, which uses up to the top-1000 centroid terms of a cluster as query.
'''
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine

import random
import nltk
import string
import re
from sklearn.cluster import KMeans
import numpy as np

from utils import initialise_llm_model

STOPWORDS = set(nltk.corpus.stopwords.words('english'))

##### 2. Create word2vec embedding model #####
def create_word2vec_model(corpus: list, save_model = False):
    '''
    Create a word2vec embeddings - trained on the input corpus.

    Args:
        corpus             : Input corpus
        save_model         : Flag for whether or not to create + save a new word2vec model. 
                             Only set to True, if no word2vec embedding has been trained. [Refer to main.py - Global parameter SAVE_EMBEDDING_MODEL]
    Returns:
        model              : Word2Vec embeddings
    '''
    
    tokenised_corpus = []
    for doc in corpus:
        words = nltk.word_tokenize(doc)
        for i, word in enumerate(words):
            words[i] = word.lower()
        tokenised_corpus.append(words)

    # Train the Word2Vec model
    model = Word2Vec(tokenised_corpus, vector_size=100, window=5, min_count=1, workers=4)

    # Save Model into directory
    if save_model:
        model.save('models/word2vec_model.bin')
        model = Word2Vec.load('models/word2vec_model.bin')
    return model

##### 3. Cluster based on embeddings #####
def process_words(words: list):
    '''
    Process a list of words, and remove words that are:
        - punctionations, stopwords (determined by nltk), doesn't contain an alphabetic character, doesn't begin with an alphabetic character

    Args:
        words             : List of words

    Returns:
        semantic_words    : List of words, that have semantic meaning.
    '''
    semantic_words = []
    for word in words:
        if word in string.punctuation:
            continue
        if word in STOPWORDS:
            continue
        if not re.search('[a-zA-Z]', word):  # checks if word contains at least one alphabetic character
            continue
        if not word[0].isalpha():            # remove words that do not begin with an alphabetic character
            continue
        semantic_words.append(word)
    
    return semantic_words

def calc_cosine(vec_a, vec_b):
    '''Calculate the cosine similarity between two vectors'''
    return 1 - cosine(vec_a, vec_b)

def get_centre_terms(cluster_centers, word_clusters, embed_model, k = 1000):
    '''
    Given a cluster center embedding value, retrieve the k closest terms, as determined by cosine similarity.

    Args:
        cluster_centers             : List of center embeddings.
        word_clusters               : List of word clusters. Each word clusters is a list of words.
        embed_model                 : The embedding model, used to retrieve the vector representation of a word.
        k                           : The number of center words to retrieve.

    Returns:
        center_terms                : List of center words. Center words are a list of k words closest to the center.
    '''
    
    center_terms = []
    for cluster_id, centroid_vector in enumerate(cluster_centers):
        cluster_words = word_clusters[cluster_id]

        # Get the top k closest terms to the centroid embedding.
        top_k_terms = []
        for word in cluster_words:
            word_vector = embed_model.wv[word]
            # Calc Cosine similarity
            dist = calc_cosine(centroid_vector, word_vector) # model.wv.similarity(word, topic)
            top_k_terms.append((word, dist))
        
        # Get Top k topics
        term_dists = sorted(top_k_terms, key=lambda x:x[1], reverse=True)
        top_k_topics = [term for term, dist in term_dists[:k]]
        # Append to cluster center terms list.
        center_terms.append(top_k_topics)

    return center_terms

def randomise_terms_selection(terms):
    '''Randomise the index slice of a given list. (Ensure there is at least 200 elements in the list)'''
    n = len(terms)
    start_idx = random.randint(0, (n//2 - 100))
    end_idx = random.randint((n//2 + 100), len(terms))
    return start_idx, end_idx

def run_batch_llm(centre_terms, parent_node, is_child):
    '''
    Extract the most representative topic from a word cluster.
    Batch process the prompts. Note: There are instances where batch processing will fail. In this case, we will reinitalise the LLM model,
    randomise the terms fed into the LLM, and try again.

    Args:
        centre_terms              : List of center words. Center words are a list of k words closest to the center.
        parent_node               : Name of the parent node.
        is_child                  : Boolean indicating whether clustering is being performed on the ROOT node or not.

    Returns:
        ai_outputs                : LLM extracted topic from word cluster.
    '''
    # 0. Setup prompt for LLM
    if is_child:
        prompt = '''Generate one possible subtopic that is relevant to a smaller branch in '{parent_node}'. You can use the following terms for ideas: {terms}. 
                    Output the succinct subtopic in lowercase.'''
    else:
        prompt = '''Extract the main topic amongst these terms: {terms}. 
                    Output topic in lowercase.{parent_node}'''
        
     # Attempt to run batch processing on ALL center terms.
    start_idx = 0
    end_idx = len(centre_terms[0]) # Number of center terms featured in the first cluster.
    while True:
        # a. Initialise a new LLM model.
        llm = initialise_llm_model()

        try:
            all_prompts = []
            for cluster_idx, terms in enumerate(centre_terms):
                all_prompts.append(prompt.format(terms = terms[start_idx:end_idx], parent_node = parent_node))
            
            # Batch Process.
            ai_outputs = llm.batch(all_prompts, config={"max_concurrency": 20})
            
            return ai_outputs
            
        except:
            print(f"Batch Process on {start_idx}:{end_idx} failed. Retrying...\nRunning batch on ")
            start_idx, end_idx = randomise_terms_selection(centre_terms[0])
            print(f"Running batch on {start_idx}:{end_idx}")


def run_clustering(word2vec_model, word_vocab, word_vectors, k_clusters = 10, parent_node = "", is_child = False):
    '''
    Extract the most representative topic from a word cluster.
    Batch process the prompts. Note: There are instances where batch processing will fail. In this case, we will reinitalise the LLM model,
    randomise the terms fed into the LLM, and try again.

    Args:
        word2vec_model            : Word2Vec embeddings
        word_vocab                : List of words
        word_vectors              : Vectors of the words
        k_clusters                : Number of clusters to initialise when running kmeans
        parent_node               : Name of the parent node.
        is_child                  : Boolean indicating whether clustering is being performed on the ROOT node or not.

    Returns:
        topic_term_clusters       : A list of tuples. Example - [(extracted_topic_from_cluster, word_cluster)]
    '''

    # 0. Setup
    semantic_words = word_vocab
    semantic_vectors = word_vectors

    # 1. Cluster setup:
    X = np.array(semantic_vectors)          # Convert word vectors to numpy array
    num_clusters = k_clusters               # Define the number of clusters

    # 1.1. Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(X)

    # 1.2. Get cluster labels and assign each word to a cluster
    cluster_labels = kmeans.labels_
    word_clusters = {}
    for word, label in zip(semantic_words, cluster_labels):
        if label not in word_clusters:                      # Not sequential numbering of clusters
            word_clusters[label] = []
        word_clusters[label].append(word)

    # 1.3. Get cluster centroid and Extract top k terms surrounding the centroid.
    cluster_centers = kmeans.cluster_centers_
    centre_terms = get_centre_terms(cluster_centers, word_clusters, word2vec_model, k = 500)

    # 1.4. Prepare LLM for batch processing -> using ONLY the top k center terms to feed to LLM.
    ai_outputs = run_batch_llm(centre_terms, parent_node, is_child)

    # 1.5. Assign relevant topic to each cluster using LLM generated results.
    topic_term_clusters = []
    llm_topics = [topic.content for topic in ai_outputs]
    for cluster_idx, terms in enumerate(centre_terms):
        # Assign entire word cluster to the topic
        cluster_of_words = word_clusters[cluster_idx]
        llm_topic = llm_topics[cluster_idx]
        topic_term_clusters.append((llm_topic, cluster_of_words))

        print(terms[:10])
        print(f"----> LLM extracted topic: {llm_topic}\n")

    return topic_term_clusters