import os
import json
import numpy as np
import pandas as pd
from pprint import pprint
from pprint import pprint as pp
from sklearn.feature_extraction.text import TfidfVectorizer
from glob2 import glob
import re
import unidecode
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

###################################
#imports needed for gsdm
from gsdmm import MovieGroupProcess
import gensim, spacy
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import Phrases
from gensim.models.phrases import Phraser
#################################
# The code for removing stopwords
stoplist = stopwords.words('english')
stoplist = set(stoplist)
question_words = ["what", "why", "when", "where", "which","would","whose","whom",
             "name", "is this", "how", "do", "does", "dont" , "is it",
             "are these", "are those", "could","should", "has", "have", "don't"]
#dir_path = r"C:\Users\pkhanwalkar\Downloads\Dataset\ds-data"
dir_path = r"/home/preeti/Desktop/Preeti-Book-Writing/Supermind-Assignment/supermind-dataset/ds-data"
def removing_stopwords(stoplist,text):
    """This function will remove stopwords which doesn't add much meaning to a sentence
       & they can be remove safely without comprimising meaning of the sentence.
   
    arguments:
         input_text: "text" of type "String".
         
    return:
        value: Text after omitted all stopwords.
       
    Example:
    Input : This is Kajal from delhi who came here to study.
    Output : ["'This", 'Kajal', 'delhi', 'came', 'study', '.', "'"]
   
   """
    # repr() function actually gives the precise information about the string
    text = repr(text)
    # Text without stopwords
    No_StopWords = [word for word in word_tokenize(text) if word.lower() not in stoplist ]
    # Convert list of tokens_without_stopwords to String type.
    words_string = ' '.join(No_StopWords)      
    return words_string
#################################


def pre_process(stplist,text):
    ###################
    #removing repeated characters and punctuations
    text=removing_stopwords(stplist,text)
    ###################
   
    ###################
    #remove punctuation
    string_punt = '!"#$%&\'()*+,-./:;<=>@[\\]^_`{|}~'
    text="".join([i for i in text if i not in string_punt])
    ###################
   
    ###################
    #remove all the occurrences of newlines, tabs, and combinations like: \\n, \\.
    text = text.replace('\\n', ' ').replace('\n', ' ').replace('\t',' ').replace('\\', ' ').replace('. com', '.com')
    ###################
   
    ###################
    # Removing all the occurrences of links that starts with https
    text = re.sub(r'http\S+', '', text)
    ###################

    ###################
    # Remove all the occurrences of text that ends with .com
    text = re.sub(r"\ [A-Za-z]*\.com", " ", text)
    text=text.lower()
    ###################

    ###################
    #remove whitespace  
    pattern = re.compile(r'\s+')
    text = re.sub(pattern, ' ', text)
    ###################

    ###################
    # There are some instances where there is no space after '?' & ')',
    # So I am replacing these with one space so that It will not consider two words as one token.
    text = text.replace('?', ' ? ').replace(')', ') ')
    ###################

    ###################
    # accented characters removal
    text = unidecode.unidecode(text)
    ###################

    return text


def collect_data(dir_path):
    #collecting all json files from the dir
    json_files = glob(dir_path+'/**/*.json') #Can be used absolute or relative paths
    print(str(len(json_files)) + ' count of json files found in ' + dir_path)
   
    #collecting all the json files in single dataframe
    dfList = []
    for jsonFile in json_files:
        df1 = pd.read_json(jsonFile)
        dfList.append(df1)    
    df = pd.concat(dfList, axis=0)
    pp(df.info())
    return df

def collect_single_data():
    #this is not needed as we have used all the json files, this was used only for testing
    p1=r"C:\Users\pkhanwalkar\Downloads\Dataset\ds-data\04-09-202220_53_24T\377801-377316.json"
    with open(p1) as f1:
        data=json.load(f1)
    df1=pd.DataFrame(data)
    return df1

def cleanup_data(df):
    #use only id and text fields
    df=df[['id','text']]
    pp(df.head())
    #explode the text: dict into multiple columns
    df=pd.concat([df.drop(['text'], axis=1), df['text'].apply(pd.Series)], axis=1)
    pp(df.head())
    #remove null entries
    pp(df[df.text.isnull()])
    df.dropna(inplace=True)
    pp(df[df.text.isnull()])
    #drop duplicate entries
    df.drop_duplicates(inplace=True)
    df.to_csv(dir_path+'/non_processed.csv')
    #df.to_csv(dir_path+'\\non_processed.csv')

    #pre process the text
    stplist = stoplist.difference(question_words)
    df['text']=df['text'].apply(lambda x:pre_process(stplist,x))
    df.to_csv(dir_path+'/processed.csv')
    #df.to_csv(dir_path+'\\processed.csv')
    return df

def get_questions(df):
    tfidf = TfidfVectorizer(max_df=.65, min_df=1, stop_words=None, use_idf=True, norm=None, vocabulary=question_words)
    features = tfidf.fit_transform(df.text).toarray()
    pp(features.shape)
    nzero=np.nonzero(features)[0]
    nz1=np.unique(nzero).tolist()
    dfq=df.iloc[nz1]
    #dfq.to_csv(dir_path+'\\q1.csv',index=True)
    dfq.to_csv(dir_path+'/q1.csv',index=True)
    return dfq


def sent_to_words(text):
    for sentence in text:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


def make_n_grams(texts):
    bigram = gensim.models.Phrases(texts, min_count=5, threshold=100)  # higher threshold fewer phrases.
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram = gensim.models.Phrases(bigram[texts], threshold=100)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    bigrams_text = [bigram_mod[doc] for doc in texts]
    trigrams_text =  [trigram_mod[bigram_mod[doc]] for doc in bigrams_text]
    return trigrams_text

# I use gensim stop-words and add me own stop-words, based on texts
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in gensim.parsing.preprocessing.STOPWORDS.union(set(['also', 'meanwhile','however', 'time','hour', 'soon', 'day', 'book', 'there', 'hotel', 'room', 'leave', 'arrive','place', 'stay', 'staff', 'location', 'service', 'come', 'check', 'ask', 'lot', 'thing', 'soooo', 'add', 'rarely', 'use', 'look', 'minute', 'bring', 'need', 'world', 'think', 'value', 'include']))] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def top_words(cluster_word_distribution, top_cluster, values):
    for cluster in top_cluster:
        sort_dicts =sorted(mgp.cluster_word_distribution[cluster].items(), key=lambda k: k[1], reverse=True)[:values]
        print("\nCluster %s : %s"%(cluster,sort_dicts))

def create_topics_dataframe(data_text,  mgp, threshold, topic_dict, lemma_text):
    result = pd.DataFrame(columns=['Text', 'Topic', 'Lemma-text'])
    for i, text in enumerate(data_text):
        result.at[i, 'Text'] = text
        #result.at[i, 'Rating'] = data.Rating[i]
        result.at[i, 'Lemma-text'] = lemma_text[i]
        prob = mgp.choose_best_label(reviews_lemmatized[i])
        if prob[1] >= threshold:
            result.at[i, 'Topic'] = topic_dict[prob[0]]
        else:
            result.at[i, 'Topic'] = 'Other'
    return result


use_q_csv = 1
use_p_csv = 1
if not use_q_csv:
    if not use_p_csv:
        df=collect_data(dir_path)
        df=cleanup_data(df)
    else:
        fn=dir_path+'/processed.csv'
        df=pd.read_csv(fn,usecols=['id', 'text'])
    dfq=get_questions(df)
else:
    fn=dir_path+'/q1.csv'
    dfq=pd.read_csv(fn,usecols=['id', 'text'])


tokens_reviews = list(sent_to_words(dfq['text']))
tokens_reviews = make_n_grams(tokens_reviews)
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# do lemmatization keeping only noun, vb, adv
# because adj is not informative for reviews topic modeling
reviews_lemmatized = lemmatization(tokens_reviews, allowed_postags=['NOUN', 'VERB', 'ADV'])

# remove stop words after lemmatization
reviews_lemmatized = remove_stopwords(reviews_lemmatized)

np.random.seed(0)

mgp = MovieGroupProcess(K=6, alpha=0.01, beta=0.01, n_iters=30)
vocab = set(x for review in reviews_lemmatized for x in review)
n_terms = len(vocab)
model = mgp.fit(reviews_lemmatized, n_terms)

doc_count = np.array(mgp.cluster_doc_count)
print('Number of documents per topic :', doc_count)

# topics sorted by the number of document they are allocated to
top_index = doc_count.argsort()[-10:][::-1]
print('\nMost important clusters (by number of docs inside):', top_index)
# show the top 5 words in term frequency for each cluster 
top_words(mgp.cluster_word_distribution, top_index, 10)


topic_dict = {}
topic_names = ['type 1','type 2', 'type 3', 'type 4','type 5','type 6']
for i, topic_num in enumerate(top_index):
    topic_dict[topic_num]=topic_names[i] 

result = create_topics_dataframe(data_text=dfq.text, mgp=mgp, threshold=0.3, topic_dict=topic_dict, lemma_text=reviews_lemmatized)
pp(result.head(5))
