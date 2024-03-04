# TopicGemini
Leveraging LLMs to improve existing Topic Modelling methods

## Project Background:
Topic Modelling is a common, yet deeply fascinating component of Natural Language Processing.
Usually, traditional approaches to this problem involve either:
    (1) Latent Dirichlet Allocation (LDA) 
    (2) Clustering and extracting centroid terms.

However, both these methods suffer from unsatisfactory topic granularity in the extracted terms,
and often, require a professional human curator to relabel the extracted 'topic' terms.

With the introduction of Large Language Models such as GPT-4 or Gemini, which are capable of incorporating semantic understanding
into natural language processing, TopicGemi explores the potential of how LLMs can be leveraged to 
improve existing processes in Topic Modelling, making them more effective for topic extraction.

## Inputs and Outputs
Input : Text Corpus                         
Output: Topic Hierarchy of Corpus.         
        - with RAG supplemented summarisations for finer granularity topic information.
