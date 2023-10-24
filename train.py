# -*- coding: utf-8 -*-
from itertools import combinations
import re

import matplotlib.pyplot as plt
from nltk import ngrams
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report,accuracy_score, plot_confusion_matrix,plot_roc_curve
import numpy as np
import pandas as pd

dataset_dictionary = None
top_word_pair_features = None
top_syntactic_grammar_list = None

trained_model_pickle_file = 'trained_model.pkl'


def get_empty_vector(n):
    return [0 for _ in range(n)]


def get_top_word_dataset_dictionary():
    from feaure_extraction.feature_vector import get_dataset_dictionary

    global dataset_dictionary
    if dataset_dictionary is None:
        dataset_dictionary = get_dataset_dictionary()
    return dataset_dictionary


def get_top_word_pair_features():
    from feaure_extraction.feature_vector import extract_top_word_pair_features

    global top_word_pair_features
    if top_word_pair_features is None:
        top_word_pair_features = extract_top_word_pair_features()
    return top_word_pair_features


def get_top_syntactic_grammar_list():
    from feaure_extraction.feature_vector import extract_top_syntactic_grammar_trio

    global top_syntactic_grammar_list
    if top_syntactic_grammar_list is None:
        top_syntactic_grammar_list = extract_top_syntactic_grammar_trio()
    return top_syntactic_grammar_list


def get_word_feature(normalized_sentence):
    listToStr = ' '.join([str(elem) for elem in normalized_sentence])
    unique_tokens = set(word for word in listToStr.split())
    # exclude duplicates in same line and sort to ensure one word is always before other
    bi_grams = set(ngrams(listToStr.split(), 2))
    words = unique_tokens | bi_grams
    dataset_dictionary = get_top_word_dataset_dictionary()
    X = [i if j in words else 0 for i, j in enumerate(dataset_dictionary)]
    return X


def get_frequent_word_pair_feature(normalized_sentence):
    listToStr = ' '.join([str(elem) for elem in normalized_sentence])
    unique_tokens = sorted(set(word for word in listToStr.split()))
    # exclude duplicates in same line and sort to ensure one word is always before other
    combos = combinations(unique_tokens, 2)
    top_word_pair_features = get_top_word_pair_features()
    X = [i if j in combos else 0 for i, j in enumerate(top_word_pair_features)]
    return X


def get_syntactic_grammar_feature(sentence_text):
    from feaure_extraction.feature_vector import extract_syntactic_grammar
    sentence_text = ' '.join([str(elem) for elem in sentence_text])
    trigrams_list = extract_syntactic_grammar(sentence_text)
    top_syntactic_grammar_list = get_top_syntactic_grammar_list()
    X = [i if j in trigrams_list else 0 for i, j in enumerate(top_syntactic_grammar_list)]
    return X



def make_feature_vector(row):
    normalized_sentence = row.normalized_sentence
    sentence = row.sentence_text
    word_feature = get_word_feature(normalized_sentence)
    frequent_word_feature = get_frequent_word_pair_feature(normalized_sentence)
    syntactic_grammar_feature = get_syntactic_grammar_feature(sentence)

    features = word_feature
    features.extend(frequent_word_feature)
    features.extend(syntactic_grammar_feature)
    return features


def main():
    from dataset.read_dataset import get_dataset_dataframe
    DIR ='D:/1_Mojtaba_/1_other_project/Matlab_DL/Project_/2_Improed_kmeans_1200/Textmining/MY_CODE/dataset/MY_DATA/'
    df = get_dataset_dataframe(DIR)
    X, Y = extract_training_data_from_dataframe(df)
    from sklearn.ensemble import RandomForestClassifier
    X_train, X_test, y_train, y_test = \
        train_test_split(X, Y, test_size=.4, random_state=0)

    print('X: ', (X.shape), 'Y : ', np.array(Y.shape))
    print(Y)
    # model = SVC(kernel='linear')
    # model.fit(X_train, y_train)
    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    print('Test Score : ', score)
    import pandas as pd

    pd.to_pickle(model, trained_model_pickle_file)
    # classification_report()
    
    
    print("Train Report:")
    y_pred_train = model.predict(X_train)
    print(classification_report(y_train, y_pred_train))
    
    print("Train accurcy:", accuracy_score(y_train, y_pred_train)*100)
    
    # plot_roc_curve(model, X_train, y_train)
    # plt.title("Roc curve for Trian data")
    # plt.show()
    
    plot_confusion_matrix(model, X_train, y_train)  
    plt.title("Confusion matrix for Trian data")
    plt.show()
    
    print("Test Report:")
    y_pred_test = model.predict(X_test)
    print(classification_report(y_test, y_pred_test))
    print("Test accurcy: ",accuracy_score(y_test, y_pred_test)*100)
    
    # plot_roc_curve(model, X_test, y_test)
    # plt.title("Roc curve for Test data")
    # plt.show()
    
    plot_confusion_matrix(model, X_test, y_test)  
    plt.title("Confusion matrix for Test data")
    plt.show()

def Make_label(relation_type):
    print(set (relation_type))
    # {nan, 'advise', 'int', 'mechanism', 'effect'}
    Label = relation_type.replace(['advise', 'int', 'mechanism', 'effect'],
                        [0, 1, 2, 3])
    
    Label = np.array(Label.tolist())
    Loc = np.squeeze(np.argwhere(np.isnan(Label)))
    Label[Loc]=1
    return Label
    

def extract_training_data_from_dataframe(df):
    X = df.apply(make_feature_vector, axis=1)
    X = np.array(X.tolist())
    relation_type = df['relation_type']
    Y = Make_label(relation_type)
    Y = np.array(Y.tolist())
    return X, Y
