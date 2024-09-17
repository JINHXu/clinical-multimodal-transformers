# -*- coding: utf-8 -*-

# import nltk
# nltk.download('stopwords')
import pandas as pd
from nltk.corpus import stopwords
import re
import nltk
from collections import Counter
import json
import pickle
import numpy as np
from tqdm import tqdm
tqdm.pandas()
# from nltk import tokenize

import spacy

# import smart_cond as sc
# from nltk.stem import SnowballStemmer
# from nltk.stem import WordNetLemmatizer

# load scispacy model
nlp = spacy.load("en_core_sci_md")

# count number of sepsis mentions and sepsis sentences
sentence_count = 0
sepsis_setence_count = 0

# clinical text preprocessing
def preprocess_text(text, nlp=nlp):
    
    # count num sentences and sepsis/septic sentences
    global sentence_count
    global sepsis_setence_count
    
    text = str(text)
    # # case normalization
    # text = text.lower().split()
    # text = " ".join(text)
    
    # avoid label leackage: remove senteces with septic or sepsis
    # sentences = tokenize.sent_tokenize(text)
    doc = nlp(text)
    spacy_sentences = list(doc.sents)
    sentences = [i.text for i in spacy_sentences]
    
    clean_sentences = []
    for sentence in sentences:
        # count number of sentences
        sentence_count += 1
        lowered_sent = sentence.lower()
        # if 'septic' in lowered_sent or 'sepsis' in lowered_sent:
        if 'SIRS' in sentence or 'sepsis' in lowered_sent:
            sepsis_setence_count += 1
            continue
        else:
            # append the org sentence, do not lowercase
            clean_sentences.append(sentence)
    
    
    text = ' '.join(clean_sentences)
    
    # special words and chars
    text = re.sub(r"yr", "year ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"don't", "do not ", text)
    text = re.sub(r"aren't", "are not ", text)
    text = re.sub(r"isn't", "is not ", text)
    text = re.sub(r"%", " percent ", text)
    text = re.sub(r"that's", "that is ", text)
    text = re.sub(r"doesn't", "does not ", text)
    text = re.sub(r"he's", "he is ", text)
    text = re.sub(r"she's", "she is ", text)
    text = re.sub(r"it's", "it is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\?", " ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    # text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    # text = re.sub(r" u s ", " american ", text)
    # # text = re.sub(r"\0s", "0", text)
    # # text = re.sub(r" 9 11 ", "911", text)
    # # text = re.sub(r"e - mail", "email", text)
    # # text = re.sub(r"j k", "jk", text)
    # # text = re.sub(r"\s{2,}", " ", text)
    # # text = text.lower().split()
    # text = [w for w in text if len(w) >= 2]
    # # text = re.sub(r" u s ", " american ", text)
    # text = re.sub(r"\0s", "0", text)
    # text = re.sub(r" 9 11 ", "911", text)
    # text = re.sub(r"e - mail", "email", text)
    # text = re.sub(r"j k", "jk", text)
    # text = re.sub(r"\s{2,}", " ", text)
    # text = text.lower().split()
    # text = [w for w in text if len(w) >= 2]

    # remove stop words
    text = text.split()
    text = [word for word in text if word not in stopwords.words('english')]
    text = " ".join(text)
    # if stemming and stops:
    #     text = [
    #         word for word in text if word not in stopwords.words('english')]
    #     wordnet_lemmatizer = WordNetLemmatizer()
    #     englishStemmer = SnowballStemmer("english", ignore_stopwords=True)
    #     text = [englishStemmer.stem(word) for word in text]
    #     text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    #     text = [
    #         word for word in text if word not in stopwords.words('english')]
    # elif stops:
    #     text = [
    #         word for word in text if word not in stopwords.words('english')]
    # elif stemming:
    #     wordnet_lemmatizer = WordNetLemmatizer()
    #     englishStemmer = SnowballStemmer("english", ignore_stopwords=True)
    #     text = [englishStemmer.stem(word) for word in text]
    #     text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    # text = " ".join(text)
    return text


# load data
data_path = '/Users/xujinghua/strats_text/data/strats_sepsis_text.pkl'
pkl = pickle.load(open(data_path, 'rb'))

data = pkl[0]
oc = pkl[1]

# # test
# text = data['value'][0]
# print(preprocess_text(text))

# text_data = data[data['variable'] == 'Text']
# texts = text_data['value'][:1]

# clean_texts = []

# for text in tqdm(texts):
#     clean_texts.append(preprocess_text(text))
    
# clean_texts = np.array(clean_texts, dtype=str)

# # preprocess text
# data.loc[data['variable'] == 'Text', 'value'] = data.loc[data['variable']
#                                                          == 'Text', 'value'].apply(preprocess_text)

# preprocess text data
data.loc[data['variable'] == 'Text', 'value'] = data.loc[data['variable'] == 'Text', 'value'].progress_apply(preprocess_text)

# NaN in text data
data.loc[data['variable'] == 'Text', 'mean'] = 1
data.loc[data['variable'] == 'Text', 'std'] = 1

# 0.0 -> 1.0 to avoid exploding gradient (NaN loss of regression model)
data.loc[data['variable'] == 'Antibiotics', 'value'] = 1
data.loc[data['variable'] == 'Blood Culture', 'value'] = 1
data.loc[data['variable'] == 'Mechanically ventilated', 'value'] = 1
# delete overlapped variable due to typo in mortality data
data = data[data['variable'] != 'vacomycin']
# drop na to avoid exploding gradient
data = data.dropna()
oc = oc.dropna()

# # alternative to dropna
# df = df[df['EPS'].notna()]

# load patient ids
train_patients_data = pd.read_csv('/Users/xujinghua/strats_text/data/pretext_large/train.csv')
train_patients = train_patients_data['patient_id'].tolist()

test_patients_data = pd.read_csv('/Users/xujinghua/strats_text/data/pretext_large/test.csv')
test_patients = test_patients_data['patient_id'].tolist()

valid_patients_data = pd.read_csv('/Users/xujinghua/strats_text/data/pretext_large/val.csv')
valid_patients = valid_patients_data['patient_id'].tolist()

# get ts_inds from ids
ids = oc['SUBJECT_ID'].tolist()

# train
train_ind = []
ts_ind = oc['ts_ind'].tolist()
for i in range(len(ts_ind)):
    if ids[i] in train_patients:
        train_ind.append(ts_ind[i])
        
        
train_ind = np.array(train_ind)

# test
test_ind = []
for i in range(len(ts_ind)):
    if ids[i] in test_patients:
        test_ind.append(ts_ind[i])
        
        
test_ind = np.array(test_ind)

# valid
valid_ind = []
for i in range(len(ts_ind)):
    if ids[i] in valid_patients:
        valid_ind.append(ts_ind[i])
        
        
# to np.array
valid_ind = np.array(valid_ind)

# dump to pkl
pickle.dump([data, oc, train_ind, valid_ind, test_ind], open(
    'sepsis_removed_2.pkl', 'wb'))

print(f'Number of sentences: {sentence_count}')
print(f'Number of septic/sepsis sentences: {sepsis_setence_count}')