
'''
Module Name: LLM_topic_refine.py
Author: James Liang
Date: 03/03/2024

After kmeans topic and word clusters are extracted...
Use LLMs to combine semantically similar clusters - by using the extracted topic terms.
'''

import copy
from utils import initialise_llm_model

##### 4. Refine topic clusters with LLM #####
def merge_topics(topic_list, term_cluster, target_topic, index_list):
    '''
    Merge topic clusters together, by using an index list to determine which clusters to merge.

    Args:
        topic_list             : List of topics
        term_cluster           : List of Cluster of terms
        target_topic           : Topic which clusters will be COMBINED INTO.
        index_list             : List of indexes, denoting clusters to be merged
    Returns:
        topic_term_cluster     : [(topic, word_clusters)...]
    '''
    
    # Merge term clusters into a single topic.
    accum_idx = index_list[0]
    temp_term_cluster = copy.deepcopy(term_cluster)

    for idx in index_list[1:]:
        temp_term_cluster[accum_idx] += temp_term_cluster[idx]

    # Create a new topic_term_cluster list that removes duplicate topics.
    temp_list = []
    for idx, topic in enumerate(topic_list):
        if topic != target_topic:
            temp_list.append((topic, temp_term_cluster[idx]))
    temp_list.append((target_topic, temp_term_cluster[accum_idx]))

    # Created a new topic_term_cluster list that removes duplicates
    topic_term_cluster = temp_list
    return topic_term_cluster

def find_indexes(lst, target_topic):
    '''Find all indexes of a target topic in a list'''
    indexes = []
    for i, value in enumerate(lst):
        if value == target_topic:
            indexes.append(i)
    return indexes

def process_same_topics(topic_term_clusters):
    '''
    Find, and merge all the clusters which have the same TOPIC assigned to them.

    Args:
        topic_term_clusters     : [(topic, word_clusters)...]

    Returns:
        topic_term_clusters     : Refined [(topic, word_clusters)...]
    '''
    # 0. Setup
    topic_list = [topic for topic, term_cluster in topic_term_clusters]
    term_cluster = [term_cluster for topic, term_cluster in topic_term_clusters]
    unique_topics = list(set(topic_list))

    # 1. Process duplicate topics
    for target_topic in unique_topics:
        while topic_list.count(target_topic) > 1:
            index_list = find_indexes(topic_list, target_topic)
            # Merge term clusters.
            topic_term_clusters = merge_topics(topic_list, term_cluster, target_topic, index_list)
            # Update the topic and term_cluster list.
            topic_list = [topic for topic, term_cluster in topic_term_clusters]
            term_cluster = [term_cluster for topic, term_cluster in topic_term_clusters]
    return topic_term_clusters

def format_llm_output(llm_output, topic_term_clusters):
    '''
    Reassign topic_term_clusters topics based on the llm output.

    Args:
        llm_output         : Output produced by LLM, featuring 'Semantic_topic': 'Original_topic'
                             For example:
                             "Psychology > attitude $ Finance > finance $ Sports > baseball"

        topic_term_clusters: [(topic, word_clusters)...]
    
    Returns:
        new_topic_term_cluster: [(semantic_topic, word_clusters)...]
    '''
    if "$" in llm_output:
        category_topics = llm_output.split("$")
    else:
        category_topics = llm_output.split("\n")
    
    new_topic_term_cluster = []
    for idx, category_topic in enumerate(category_topics):
        new_topic, prev_topic = category_topic.split(">")
        prev_topic = prev_topic.lower().strip()

        # if prev_topic == topic_term_clusters[idx][0]: # Check condition upholds. 

        new_topic = new_topic.lower().strip()

        term_clusters = topic_term_clusters[idx][1]
        new_topic_term_cluster.append((new_topic, term_clusters))

    return new_topic_term_cluster


def refine_topic_clusters(topic_term_clusters, node_name = "", is_child = False):
    '''
    Refine Topic clusters by merging semantically similar clusters.

    Args:
        topic_term_clusters : [(topic, word_clusters)...]
        node_name           : Node name of the parent.
        is_child            : Flag indicating whether we are working on the ROOT node or not.
    
    Returns:
        new_topic_term_cluster : Refined topic clusters. Example - [(topic, word_clusters)...]
    '''

    prompt = ''' {parent_topic}
                Generate the most representative category that categories each one of the topics from the following list: {llm_generated_topics}. 

                Format your output in the following way:
                representative_category > topic1 $ representative_category > topic2 $ ...
    '''
    
    # a. Initialise a new LLM model.
    llm = initialise_llm_model()

    # 1. Merge topic clusters that have been assigned the same topic.
    new_topic_term_cluster = process_same_topics(topic_term_clusters)
    topic_list = [topic for topic, term_cluster in new_topic_term_cluster]

    if not is_child: # Ie. is_root
        # 2. Merge semantically similar clusters - as determined by LLM.
        llm_output = llm.invoke(prompt.format(llm_generated_topics = topic_list, parent_topic = node_name)).content
        print(f"LLM topic refinement: {llm_output}\n")
        
        #   a. Format Output.
        new_topic_term_cluster = format_llm_output(llm_output, new_topic_term_cluster)
        #   b. Merge topic clusters with the same topic
        new_topic_term_cluster = process_same_topics(new_topic_term_cluster)
        topic_list = [topic for topic, term_cluster in new_topic_term_cluster]
    
    print(f"===> Generated Topics for this level: {topic_list}\n")
    return new_topic_term_cluster
