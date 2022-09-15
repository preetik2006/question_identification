# questions_tags_identification

Crypto data set downloaded from the given location into  local folder
https://drive.google.com/drive/u/0/folders/18Ds1aTobVbvrXGzNpR9EIAShJ_H0KSSL 

Function collect_data collects the data from the given local directory containing all the existing json file and returns a dataframe

Function collect_single_data is used to test the code for smaller set of data

Function cleanup_data does basic cleaup on dataframe by considering only text and id columns. Further, the text field which is dictionary object is broken down into multiple columns and removes duplicate and null entries. Later it does text preprocessing like converting to lower case, remove stop words, whitespaces, html tags, etc.
The resulting dataframe saved as processed.csv

Function get_questions invoked on the preprocessed dataframe which identifies question based on  tfidf vectoriser with vocabulary set to question words

This function results into a dataframe containing text determined as questions

From the given references we tried gsdmm model found from 
https://www.kaggle.com/code/ptfrwrd/topic-modeling-guide-gsdm-lda-lsi


