import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from gensim.parsing.preprocessing import remove_stopwords
import nltk.data
import logging


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)



print("Done")