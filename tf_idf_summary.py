import nltk
#nltk.download('stopwords')
#nltk.download('punkt')
import os
import re
import math
import operator
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
Stopwords = set(stopwords.words('english'))
wordlemmatizer = WordNetLemmatizer()


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
ps = PorterStemmer()


with open('articles/article_1.txt', 'r') as file:
    ARTICLE = file.read()

def text_preprocessing(sentences):
    stop_words = set(stopwords.words('english'))
    clean_words = []
    for sent in sentences:
        words = word_tokenize(sent)
        words = [ps.stem(word.lower()) for word in words if word.isalnum()]
        clean_words += [word for word in words if word not in stop_words]
    return clean_words
    
    

import math
from collections import Counter

def split_sentences(text):
    text = text.replace('\n', ' ')
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return sentences

def calculate_tf(term, document):
    term_frequency = Counter(document.split())[term]
    return term_frequency / len(document.split())

def calculate_idf(term, document_set):
    document_frequency = sum([1 for doc in document_set if term in doc])
    return math.log(len(document_set) / document_frequency) + 1

def tfidf_vectorizer(sentences, document_set):
    terms = set(term for sentence in sentences for term in sentence.split())
    terms_list = list(terms)
    term_index = {term: index for index, term in enumerate(terms_list)}

    tfidf_matrix = np.zeros((len(sentences), len(terms_list)))

    for i, sentence in enumerate(sentences):
        for term in sentence.split():
            tfidf_matrix[i, term_index[term]] = calculate_tf(term, sentence) * calculate_idf(term, document_set)

    return tfidf_matrix


def calculate_sentence_similarity(tfidf_matrix):
    normalized_matrix = tfidf_matrix / np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
    sentence_similarity_matrix = np.dot(normalized_matrix, normalized_matrix.T)
    return sentence_similarity_matrix



def calculate_sentence_scores(sentence_similarity_matrix):
    sentence_scores = np.sum(sentence_similarity_matrix, axis=1) - np.diag(sentence_similarity_matrix)
    return sentence_scores
    
    
    
import numpy as np
def generate_summary(text, num_sentences=5):
    sentences = split_sentences(text)
    tfidf_matrix = tfidf_vectorizer(sentences, sentences)
    sentence_similarity_matrix = calculate_sentence_similarity(tfidf_matrix)
    sentence_scores = calculate_sentence_scores(sentence_similarity_matrix)

    ranked_sentences = [sentence for _, sentence in sorted(zip(sentence_scores, sentences), reverse=True)]

    #print(ranked_sentences)

    summary = '. \n'.join(ranked_sentences[:num_sentences]) + '.'

    return summary

summary = generate_summary(ARTICLE, num_sentences=10)

print(summary)

