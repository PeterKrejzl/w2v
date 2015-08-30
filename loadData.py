import pandas as pd
import logging
import os
from gensim.models import Word2Vec



'''
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from gensim.parsing.preprocessing import remove_stopwords
import nltk.data

'''

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

#/Users/pk/Documents/workspace/V2W/wiki/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled
#news.en-00001-of-00100


def process_single_file(path, file):
    p = path.join(file)
    sentences = []
    print("Processing file = %s" % file)
    data = pd.read_csv(path + file, header=None, quoting=3, encoding="utf_8", delimiter="\t")
    data.columns = ['sentence']
    
    for sentence in data['sentence']:
        if len(sentence) > 0:
            words = sentence.lower().split()
            sentences.append(words)

    return(sentences)

    




total_data = []

path = '/Users/pk/Documents/workspace/V2W/wiki/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/'



num_features = 500    # Word vector dimensionality                      
min_word_count = 1   # Minimum word count                        
num_workers = 16       # Number of threads to run in parallel
context = 15          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

print ("Training model...")

model_name = "500features_1minwords_15context"


files_processed = 0
for file in os.listdir(path):
    if files_processed == 0:
        model = Word2Vec(process_single_file(path, file), size = num_features, workers=num_workers, min_count=min_word_count, window=context, sample=downsampling)
        #model.save(model_name)
        files_processed += 1
    else:
        #total_data += process_single_file(path, file)
        #model = Word2Vec.load(model_name)
        model.train(process_single_file(path, file))
        model.save(model_name)
        files_processed += 1


model.save(model_name)


#print("Data size = %d\n\n" % len(total_data))

#collected 168624 word types from a corpus of 7769690 raw words and 306068 sentences
#collected 168618 word types from a corpus of 7771908 raw words and 306362 sentences

#print(total_data[0])
#print(total_data[1])




#print("\n\n\nNumber of files to process = %d" %( len(os.listdir(path))))
#print("\n\n\n")


model.init_sims(replace=True)
model.save(model_name)




print("Done")


