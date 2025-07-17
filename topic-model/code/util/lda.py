import numpy as np
import random

import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.test.utils import datapath

from nltk.corpus import stopwords

def get_lda_corpus(init_docs_proc):
    # Create a set of frequent words
    stoplist = stopwords.words('english')
    stoplist.extend(['organization', 'subject', 'lines', 'unk'])

    # Lowercase each document, split it by white space and filter out stopwords
    texts = [[word for word in document.lower().split() if word not in stoplist]
            for document in init_docs_proc]

    # Count word frequencies
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    # Only keep words that appear more than once
    processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]

    dictionary = corpora.Dictionary(processed_corpus)

    for key in dictionary.token2id:
        dictionary.id2token[dictionary.token2id[key]] = key

    lda_corpus = [dictionary.doc2bow(text) for text in processed_corpus]

    return lda_corpus, dictionary

def get_lda_models(lda_docs, N_t, passes=10, debug=False):

    lda_corpus, dictionary = get_lda_corpus(lda_docs)
    iterations = 200

    print('-- Training LDA --')

    lda_models = []
    seeds = [1000,2000,3000]

    if debug:
        seeds = [1000,2000,3000]
        passes = 1
        iterations = 2

    for seed in seeds:
        np.random.seed(seed)
        random.seed(seed)

        lda = LdaModel(lda_corpus, num_topics=20, iterations=iterations, passes=passes)
        print('LDA lower bound', lda.log_perplexity(lda_corpus))
        lda_models.append(lda)

    print('-- Printing topics --')
    for i in range(20):
        word_list = []
        pairs = lda.get_topic_terms(i, topn=15)
        for pair in pairs:
            word_list.append(dictionary.id2token[pair[0]])
        print(word_list)
    print('----------------------')

    lda_corpus_t = lda_corpus[:N_t]
    lda_corpus_v = lda_corpus[N_t:]

    return lda_models, lda_corpus_t, lda_corpus_v