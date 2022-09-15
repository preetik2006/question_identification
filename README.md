# questions_tags_identification

Crypto data set downloaded from the given location into  local folder
https://drive.google.com/drive/u/0/folders/18Ds1aTobVbvrXGzNpR9EIAShJ_H0KSSL 

Function collect_data collects the data from the given local directory containing all the existing json files and returns a data frame

Function collect_single_data is used to test the code for a smaller set of data

Function cleanup_data does basic clean-up on the data frame by considering only text and id columns. Further, the text field which is a dictionary object is broken down into multiple columns and removes duplicate and null entries. Later it does text preprocessing like converting to lower case, removing stop words, whitespaces, Html tags, etc.
The resulting data frame saved as processed.csv

Function get_questions invoked on the preprocessed data frame which identifies questions based on  TF-IDF vectorizer with vocabulary set to question words

This function results in a data frame containing text determined as questions

From the given references we tried gsdmm model found on the link given below and followed the outlined steps.
https://www.kaggle.com/code/ptfrwrd/topic-modeling-guide-gsdm-lda-lsi

Tokenization and delete punctuation step is followed using function sent_to_words

N-grams were created using make_n_grams

remove_stopwords is used to remove stop words, where gensim stop-words are used and add our own stop-words, are added

lemmatization function is used to identify nouns, adjectives, verb,s and adverbs 

Further, we use GSDMM topic modeling with different hyperparameters which divides questions into different clusters and identifies top words per cluster


Further, we created a data frame with text columns from the original data frame, lemmatized text, and best tags






