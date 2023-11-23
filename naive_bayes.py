# EECS 487 Intro to NLP
# Assignment 1

import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from math import sqrt

should_add_k_for_unseen = True


def load_headlines(filename):
    df = pd.read_csv(filename)
    df = df[['ratings', 'reviews']]
    df['reviews'].replace(np.nan, "", inplace=True)
    return df

def get_basic_stats(df):
    avg_len = 0
    std_len = 0
    num_articles = {0: 0, 1: 0}
    
    for i in range(len(df)):
        if df.iloc[i]['ratings'] < 4:
            num_articles[0] = num_articles[0] + 1
        else:
            num_articles[1] = num_articles[1] + 1
        review = df.iloc[i]['reviews']
        text = review.lower()
        tokens = word_tokenize(text)
        avg_len = avg_len + len(tokens)
    avg_len = avg_len / len(df)
    
    std_dev = 0
    for text in df['reviews']:
        text = text.lower()
        tokens = word_tokenize(text)
        num = len(tokens) - avg_len 
        num = num*num
        std_dev = std_dev + num
    std_dev = std_dev / len(df)
    std_len = sqrt(std_dev)

    print(f"Average number of tokens per headline: {avg_len}")
    print(f"Standard deviation: {std_len}")
    print(f"Number of negative/positive headlines: {num_articles}")
    
    return num_articles
    
def get_baseline(df):
    num_articles = get_basic_stats(df)
    majority_class = ""
    if(num_articles[0] > num_articles[1]):
        majority_class = "negative"
    else:
        majority_class = "positive"
    return majority_class

def get_baseline_perf(df):
    #For a majority class baseline, we assume a poorly designed model always predicts the majority class
    #Our classifier should perform better than a model that's blind to minority classes
    num_articles = get_basic_stats(df)
    return max(num_articles[0],num_articles[1]) / (num_articles[0] + num_articles[1]) 


class NaiveBayes:
    """Naive Bayes classifier."""

    def __init__(self):
        self.ngram_count = []
        self.total_count = []
        self.category_prob = []
        self.vectorizer = CountVectorizer(ngram_range = (1,2), min_df = 3, max_df = 0.8, lowercase = True)
    
    def fit(self, data):
        transform_vocab = self.vectorizer.fit_transform(data['reviews']) # this gives array of count of all words
        transform_vocab = transform_vocab.toarray()
        self.ngram_count = np.zeros((2, len(transform_vocab[0])))
        self.total_count = np.zeros(2)
        training_labels = np.zeros(len(data))

        for i in range(len(data)):
            if data.iloc[i]['ratings'] < 4:
                training_labels[i] = 0
            else:
                training_labels[i] = 1
        
        self.category_prob = np.zeros(2)
        labels_sum = training_labels.sum()
        self.category_prob[1] = labels_sum / len(data)
        self.category_prob[0] = (len(data) - labels_sum) / len(data)
                                   
        for c in range(2):
            w_c = transform_vocab[training_labels == c]
            w_c_total = w_c.sum() # gives count(w,c) for all w with label c in V
            self.total_count[c] = w_c_total
            w_c_count = w_c.sum(axis=0) # get count for each word with c-label
            self.ngram_count[c] = (np.array(w_c_count))


        ###################################################################
    
    def calculate_prob(self, docs, c_i):
        prob = np.zeros(len(docs))
        loglikelihood = np.zeros(len(self.ngram_count[c_i]))
        
        denom = self.total_count[c_i] + len(self.ngram_count[c_i])
        numer = self.ngram_count[c_i] + 1
        val = numer / denom
        loglikelihood = np.log(val) # log likelihoods for each word in vocab
        
        test_transform = self.vectorizer.transform(docs)
        test_transform = test_transform.toarray()
        
        test_likelihoods = test_transform * loglikelihood
        test_likelihoods = test_likelihoods.sum(axis=1)
        
        #test_likelihoods = test_likelihoods.sum()
        
        prob = np.log(self.category_prob[c_i]) + test_likelihoods

        return prob

    def predict(self, docs):
        prediction = [None] * len(docs)

        pred0 = self.calculate_prob(docs, 0)
        pred1 = self.calculate_prob(docs, 1)
        
        for i in range(len(docs)):
            if pred0[i] > pred1[i]:
                prediction[i] = 0
            else:
                prediction[i] = 1
        return prediction


def evaluate(predictions, labels):
    accuracy, mac_f1, mic_f1 = None, None, None

    ###################################################################
    # TODO: calculate accuracy, macro f1, micro f1
    # Note: you can assume labels contain all values from 0 to C - 1, where
    # C is the number of categories
    ###################################################################

    ###################################################################
    #accuracy = true positives / all positives
    #f1 = 2pr/p+r
    
    tp0 = 0
    tn0 = 0
    fp0 = 0
    fn0 = 0
    tp1 = 0
    tn1 = 0
    fp1 = 0
    fn1 = 0

    for i in range(len(predictions)):
        if predictions[i] == 0:
            if labels[i] == predictions[i]:
                tp0 = tp0 + 1
                tn1 = tn1 + 1
            else:
                fp0 = fp0 + 1
                fn1 = fn1 + 1

        else:
            if labels[i] == predictions[i]:
                tp1 = tp1 +1
                tn0 = tn0 + 1
            else:
                fn0 = fn0 + 1
                fp1 = fp1 + 1
   
    accuracy = (tp0 + tp1) / (tp0 + tp1 + fp0 + fp1)
    
    precision0 = tp0 / (tp0 + fp0)
    recall0 = tp0 / (tp0 + fn0)
    precision1 = tp1 / (tp1 + fp1)
    recall1 = tp1 / (tp1 + fn1)
    f1_0 = (2 * precision0 * recall0) / (precision0 + recall0)
    f1_1 = (2 * precision1 * recall1) / (precision1 + recall1) 
    
    mac_f1 = (f1_0 + f1_1) / 2
    
    precision_micro = (tp0 + tp1) / (tp0 + tp1 + fp0 + fp1)
    recall_micro = (tp0 + tp1) / (tp0 + tp1 + fn0 + fn1)
    
    mic_f1 = (2 * precision_micro * recall_micro) / (precision_micro + recall_micro)
    
    
    return accuracy, mac_f1, mic_f1
