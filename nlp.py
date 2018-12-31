import pandas as pd
import numpy as np



import spacy
spacy.load('en')
import nltk
#nltk.download("wordnet")
from spacy.lang.en import English
parser = English()
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora
import gensim


stop_words_en=gensim.parsing.preprocessing.STOPWORDS





def tokenize(text):
	"""
	transforme le texte en tokens
	"""
	lda_tokens = []
	tokens = parser(text)
	for token in tokens:
		if token.orth_.isspace():
			continue
		elif token.like_url:
			lda_tokens.append('URL')
		elif token.orth_.startswith('@'):
			lda_tokens.append('SCREEN_NAME')
		else:
			lda_tokens.append(token.lower_)
	return lda_tokens

def get_lemma(word):
	"""
	lemmatization des mots (pour le NLP)
	"""
	lemma = wn.morphy(word)
	if lemma is None:
		return word
	else:
		return lemma


def prepare_text_for_lda(text):
	"""
	Préparation compète du texte pour le NLP
	"""
	tokens = tokenize(text.replace("-"," "))
	tokens = [token for token in tokens if len(token) > 4]
	tokens = [token for token in tokens if token not in stop_words_en]
	tokens = [get_lemma(token) for token in tokens]
	return tokens

def texttokens(texts):
	"""main.py
	Préparation de tous les textes pour le NLP
	"""
	text_data = []
	for t in texts:
		tokens = prepare_text_for_lda(t)
		text_data.append(tokens)
	return text_data








def LSI_topicExtraction(texts, n_topics):
	"""
	Extraction des topics avec l'algorithme LDA
	"""


	print("Tokenization...")
	text_data=texttokens(texts)
	print("Dictionarisation...")
	dictionary = corpora.Dictionary(text_data)
	print("Corpusisation...")
	corpus = [dictionary.doc2bow(text) for text in text_data]
    

	#print(corpus)
	print("modelization...")
	lsimodel = gensim.models.LsiModel(corpus, id2word=dictionary,num_topics=n_topics)

	return lsimodel, corpus


