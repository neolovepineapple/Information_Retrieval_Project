import itertools
import re
from collections import Counter, defaultdict
from typing import Dict, List, NamedTuple

import numpy as np
from numpy.linalg import norm
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split

import math
import argparse
import json

### File IO and processing

class Document(NamedTuple):
    title: List[str]
    author: List[str]
    content: List[str]

    def sections(self):
        return [self.title, self.author, self.content]

    def __repr__(self):
        return (f"title: {self.title}\n" +
            f"  author: {self.author}\n" +
            f"  content: {self.content}")


def read_stopwords(file):
    with open(file) as f:
        return set([x.strip() for x in f.readlines()])

stopwords = read_stopwords('common_words')

stemmer = SnowballStemmer('english')

def read_docs(file, key_type=0, limit=None):
    key_types = [
        ['title', 'content'],
        ['headline', 'short_description'],
    ]
    keys = key_types[key_type]
    docs = []
    
    cnt = 0
    with open(file) as f:
        for line in f:
            cnt += 1
            if limit and cnt > limit:
                break
            line = json.loads(line.strip())
            
            words_list = []
            for key in keys:
                words_list.append([ word.lower() for word in word_tokenize(line[key]) ])
            docs.append(Document(words_list[0], [], words_list[1]))

    return docs

def stem_doc(doc: Document):
    return Document(*[[stemmer.stem(word) for word in sec]
        for sec in doc.sections()])

def stem_docs(docs: List[Document]):
    return [stem_doc(doc) for doc in docs]

def remove_stopwords_doc(doc: Document):
    return Document(*[[word for word in sec if word not in stopwords]
        for sec in doc.sections()])

def remove_stopwords(docs: List[Document]):
    return [remove_stopwords_doc(doc) for doc in docs]



### Term-Document Matrix

class TermWeights(NamedTuple):
    title: float
    author: float
    content: float

def compute_doc_freqs(docs: List[Document]):
    '''
    Computes document frequency, i.e. how many documents contain a specific word
    '''
    freq = Counter()
    for doc in docs:
        words = set()
        for sec in doc.sections():
            for word in sec:
                words.add(word)
        for word in words:
            freq[word] += 1
    return freq

def compute_tf(doc: Document, doc_freqs: Dict[str, int], weights: list, n_docs):
    vec = defaultdict(float)
    for word in doc.title:
        vec[word] += weights.title
    for word in doc.author:
        vec[word] += weights.author
    for word in doc.content:
        vec[word] += weights.content
    return dict(vec)  # convert back to a regular dict

def compute_tfidf(doc, doc_freqs, weights, n_docs):
    vec = defaultdict(float)
    tf = compute_tf(doc, doc_freqs, weights, n_docs)
    for word in tf:
        if doc_freqs[word] == 0: continue
        # vec[word] = (1 + math.log(tf[word])) * math.log(N/doc_freqs[word])
        vec[word] = tf[word] * math.log(n_docs/doc_freqs[word])
    return dict(vec)

def compute_boolean(doc, doc_freqs, weights, n_docs):
    vec = defaultdict(float)
    for word in doc.title:
        vec[word] = max(vec[word], weights.title)
    for word in doc.author:
        vec[word] = max(vec[word], weights.author)
    for word in doc.content:
        vec[word] = max(vec[word], weights.content)
    return dict(vec)



### Vector Similarity

def dictdot(x: Dict[str, float], y: Dict[str, float]):
    '''
    Computes the dot product of vectors x and y, represented as sparse dictionaries.
    '''
    keys = list(x.keys()) if len(x) < len(y) else list(y.keys())
    return sum(x.get(key, 0) * y.get(key, 0) for key in keys)

def cosine_sim(x, y):
    '''
    Computes the cosine similarity between two sparse term vectors represented as dictionaries.
    '''
    num = dictdot(x, y)
    if num == 0:
        return 0
    return num / (norm(list(x.values())) * norm(list(y.values())))

def dice_sim(x, y):
    num = dictdot(x, y)
    if num == 0: return 0
    de = sum(x.values()) + sum(y.values())
    return 2 * num / de

def jaccard_sim(x, y):
    num = dictdot(x, y)
    de = sum(x.values()) + sum(y.values()) - num
    if de == 0: return float("inf")
    return num / de

def overlap_sim(x, y):
    num = dictdot(x, y)
    if num == 0: return 0
    de = min(sum(x.values()), sum(y.values()))
    return num / de



### Search

def prepare_data():
    X1 = read_docs('CNN_output.json') + read_docs('article_cdc.json') + read_docs('output_WP.json') + read_docs('output_BBC.json')
    y1 = np.zeros(len(X1))
    X2 = read_docs('News_Category_Dataset_v2.json', key_type=1, limit=1500)
    y2 = np.ones(len(X2))

    X = X1 + X2
    y = np.concatenate((y1, y2))
    return X, y


stopwords = read_stopwords('common_words')

term_funcs = {
    'tf': compute_tf,
    'tfidf': compute_tfidf,
    # 'boolean': compute_boolean
}

sim_funcs = {
    'cosine': cosine_sim,
    # 'jaccard': jaccard_sim,
    'dice': dice_sim,
    'overlap': overlap_sim
}

permutations = [
    term_funcs,
    # [True],  # stem
    # [True],  # remove stopwords
    [False, True],  # stem
    [False, True],  # remove stopwords
    [
        TermWeights(title=1, author=1, content=1),
        TermWeights(title=3, author=1, content=1),
        # TermWeights(title=1, author=1, content=3),
        ]
]

term = 'tfidf'
sim = 'cosine'

def gen_vector(docs, term, stem, removestop, term_weights):
    processed_docs = process_docs(docs, stem, removestop, args)

    doc_freqs = compute_doc_freqs(processed_docs)
    n_docs = len(docs)
    doc_vectors = [term_funcs[term](doc, doc_freqs, term_weights, n_docs) for doc in processed_docs]
    return doc_vectors
    
def gen_profile(doc_vectors):
    N = len(doc_vectors)
    vec = defaultdict(float)
    for doc_vector in doc_vectors:
        for word in doc_vector:
            vec[word] += doc_vector[word]/N
    return vec


def experiment(args):
    X, y = prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    print('term', 'stem', 'removestop', 'termweights', 'accuracy', sep='\t')

    for term, stem, removestop, term_weights in itertools.product(*permutations):
        X_train1 = list(x for i, x in enumerate(X_train) if y_train[i] == 0)
        X_train2 = list(x for i, x in enumerate(X_train) if y_train[i] == 1)

        train_vectors1 = gen_vector(X_train1, term, stem, removestop, term_weights)
        train_vectors2 = gen_vector(X_train2, term, stem, removestop, term_weights)

        profile1 = gen_profile(train_vectors1)
        profile2 = gen_profile(train_vectors2)

        test_vectors = gen_vector(X_test, term, stem, removestop, term_weights)

        def predict(v):
            sim1 = sim_funcs[sim](v, profile1)
            sim2 = sim_funcs[sim](v, profile2)
            diff = sim1-sim2
            return (0 if diff >= 0 else 1, diff, sim1, sim2)

        predictions = np.array([predict(v)[0] for v in test_vectors])
        correct_count = sum(y_test == predictions)
        acc = correct_count/len(y_test)
        
        print(term, stem, removestop, ','.join(map(str, term_weights)), acc, sep='\t')

def process_docs(docs, stem, removestop, args):
    processed_docs = docs
    if removestop:
        processed_docs = remove_stopwords(processed_docs)
    if stem:
        processed_docs = stem_docs(processed_docs)

    return processed_docs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    experiment(args)