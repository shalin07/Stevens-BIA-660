
# coding: utf-8

# In[4]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import re
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from scipy.spatial.distance import cosine
from nltk.probability import FreqDist
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from scipy.spatial import distance
from termcolor import colored

import spacy
import en_core_web_sm 
nlp = en_core_web_sm.load()

def extract(text):
    return re.findall('([A-Z]{1}[a-z]+ [A-Z]{1}[a-z]+),([ a-zA-Z]+[a-z]{1})[ ,(a-z]+(2[0-9]{3})[): $]+([0-9]{3},[0-9]{3})', text)
text='''Following is total compensation for other presidents at privat
e colleges in Ohio in 2015:
Grant Cornwell, College of Wooster (left in 2015): $911,651
Marvin Krislov, Oberlin College (left in 2016): $829,913
Mark Roosevelt, Antioch College, (left in 2015): $507,672
Laurie Joyner, Wittenberg University (left in 2015): $463,504
Richard Giese, University of Mount Union (left in 2015): $453,800'''

def tokenize(text, lemmatized = False, no_stopword = False):
    tokens = []
    text = nlp(text)
    if lemmatized:
        if no_stopword:
            tokens = [token.lemma_ for token in text if not token.is_stop]
        else:
            tokens = [token.lemma_ for token in text]
    if lemmatized == False:
        if no_stopword:
            tokens = [token.text for token in text if not token.is_stop]
        else:
            tokens = [token.text for token in text]
    token_count = FreqDist(tokens)
    return token_count

def get_similarity(q1, q2, lemmatized=False, no_stopword=False):
    a = lemmatized
    b = no_stopword
    sim = []
    q1 = q1 + q2
    for i in q1:
        q3 = [''.join(i) for i in q1]
    docs_tokens={idx:tokenize(doc,a,b) for idx,doc in enumerate(q3)}
    dtm=pd.DataFrame.from_dict(docs_tokens, orient="index" )
    dtm=dtm.fillna(0)
      
    # step 4. get normalized term frequency (tf) matrix        
    tf=dtm.values
    doc_len=tf.sum(axis=1)
    tf=np.divide(tf.T, doc_len).T
    
    # step 5. get idf
    df=np.where(tf>0,1,0)
    #idf=np.log(np.divide(len(docs), \
    #    np.sum(df, axis=0)))+1

    smoothed_idf=np.log(np.divide(len(q3)+1, np.sum(df, axis=0)+1))+1    
    smoothed_tf_idf=tf*smoothed_idf
    
    similarity=1-distance.squareform (distance.pdist(smoothed_tf_idf, 'cosine'))
    for i in range(0,500):
        sim.append(similarity[i,i+500])
    return sim

data = pd.read_csv("quora_duplicate_question_500.csv")
q1 = data["q1"].values.tolist()
q2 = data["q2"].values.tolist()
ground_truth = data["is_duplicate"].values
def predict(sim, ground_truth, threshold = 0.5):
    count = 0
    count1 = 0
    count2 = 0
    for i in range(0,500):
        if sim[i] > threshold:
            sim[i] = 1
        else:
            sim[i] = 0
    pred = sim
    for i in range(500):
        if sim[i] == ground_truth[i] == 1:
            count += 1
    for i in range(500):
        if sim[i]==1:
            count2 += 1
    count1=data['q1'].loc[(data.is_duplicate == 1.0)].count()
    recall = count/count1
    
    return pred, recall

def evaluate(sim, ground_truth, threshold = 0.5 ):
    count = 0
    count1 = 0
    count2 = 0
    for i in range(0,500):
        if sim[i] > threshold:
            sim[i] = 1
        else:
            sim[i] = 0
    for i in range(500):
        if sim[i] == ground_truth[i] == 1:
            count += 1
    for i in range(500):
        if sim[i]==1:
            count2 += 1
    count1=data['q1'].loc[(data.is_duplicate == 1.0)].count()
    recall = count/count1
    precision = count/count2
    return recall, precision



print(colored("Output of Question 1:", "blue", attrs=['bold']))
extract(text)

print(colored("Test Q1", "green", attrs=['bold']))
print("\nlemmatized: False, no_stopword: False")
sim = get_similarity(q1, q2, lemmatized=False, no_stopword=False)
pred, recall = predict(sim, ground_truth)
print(recall)
print('')


print(colored("Test Q2", "green", attrs=['bold']))
print("\nlemmatized: True, no_stopword: False")
sim = get_similarity(q1, q2, lemmatized=True, no_stopword=False)
pred, recall = predict(sim, ground_truth)
print(recall)
print('')


print(colored("Test Q3", "green", attrs=['bold']))
print("\nlemmatized: False, no_stopword: True")
sim = get_similarity(q1, q2, lemmatized=False, no_stopword=True)
pred, recall = predict(sim, ground_truth)
print(recall)
print('')

print(colored("Test Q4", "green", attrs=['bold']))
print("\nlemmatized: True, no_stopword: True")
sim = get_similarity(q1, q2, lemmatized=True, no_stopword=True)
pred, recall = predict(sim, ground_truth)
print(recall)
print('')

print(colored("Output of Q3", "green", attrs=['bold']))
recall, precision = evaluate(sim, ground_truth, threshold = 0.5 )
print(recall, precision)

