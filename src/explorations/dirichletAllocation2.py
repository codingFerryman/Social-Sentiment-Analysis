# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import datetime as dt
import enum
import json
import os
import sys
from pathlib import Path
from typing import Tuple
import gensim
import gensim.corpora as corpora
from gensim.test.utils import common_corpus, common_dictionary
import pyLDAvis
import pickle
from pyLDAvis import gensim as gensimvis

import nltk
from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from dateutil import parser

import hyperopt
import hyperopt.pyll
import numpy as np
from datasets import list_metrics
from transformers import logging as hf_logging

sys.path.append(os.path.join(os.getcwd(), '..'))
from utils import get_project_path, get_transformers_layers_num, loggers
from models.Model import ModelConstruction
from models.transformersModel import TransformersModel
from preprocessing.cleaningText import cleaningMap
from preprocessing.pretrainedTransformersPipeline import PretrainedTransformersPipeLine

PROJECT_DIRECTORY = get_project_path()

# hf_logging.set_verbosity_error()
hf_logging.enable_explicit_format()
logger = loggers.getLogger("Notebook", debug=True)

def main():
    # %%
    d = {}
    with open(os.path.join(PROJECT_DIRECTORY, "src/configs/dev/squeezebert.json")) as fr:
        d = json.load(fr)


    # %%
    model_name_or_path = d['model_name_or_path']
    tokenizer_name_or_path = d.get('tokenizer_name_or_path', model_name_or_path)
    model = TransformersModel(modelName_or_pipeLine=model_name_or_path,
                                tokenizer_name_or_path=tokenizer_name_or_path,
                                fast_tokenizer=d.get('fast_tokenizer'),
                                text_pre_cleaning=d.get('text_pre_cleaning', 'default'))


    # %%
    if type(d['metric']) is str:
            d['metric'] = [d['metric']]
    assert (d['metric'][0] in list_metrics()),         f"The metric for evaluation is not supported.\n"         f"It should be in https://huggingface.co/metrics"


    # %%
    model.registerMetric(*d['metric'])


    # %%
    model.loadData(ratio=d['data_load_ratio'])


    # %%
    text_pre_cleaning_function = cleaningMap("tweet")
    logger.info('Cleaning the dataset ...')
    allData = text_pre_cleaning_function(model.pipeLine.allData)

    np.save('allDatacleanedText', allData)
    # %%
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

    def lemmatize(input):
        """
        Lemmatizes input using NLTK's WordNetLemmatizer
        """
        lemmatizer=WordNetLemmatizer()
        input_str=word_tokenize(input)
        new_words = []
        for word in input_str:
            new_words.append(lemmatizer.lemmatize(word))
        return ' '.join(new_words)


    # %%
    tt = TweetTokenizer()


    # %%
    dataTokenized = [tt.tokenize(dat) for dat in allData]


    # %%
    # Create Dictionary
    id2word = corpora.Dictionary(dataTokenized)
    # Create Corpus
    corpus = [id2word.doc2bow(text) for text in dataTokenized]

    # %%
    num_topics = 10 # positive/negative
    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                        id2word=id2word,
                                        num_topics=num_topics)

    # %%
    with open("../../data/test_data.txt") as fr:
        testTweets = fr.readlines()
    testTweetsTokenized = [tt.tokenize(testTweet) for testTweet in testTweets]

    # %%
    testTweetBow = [common_dictionary.doc2bow(tweet) for tweet in testTweetsTokenized]


    # %%
    testTweetsVectors = np.array([lda_model[it] for it in testTweetBow])


    # %%
    vectorProbabilities = testTweetsVectors[:,:,1]
    test_topics = np.argmax(vectorProbabilities, axis=-1)
    print(test_topics)
    # print()


    # %%
    # from pyLDAvis import gensim_models 
    LDAvis_data_filepath = os.path.join(f'./ldavis_prepared2_{num_topics}.ld')

    ## This is a bit time consuming - make the if statement True
    ## if you want to execute visualization prep yourself
    if 1 == 1:
        LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
        with open(LDAvis_data_filepath, 'wb') as f:
            pickle.dump(LDAvis_prepared, f)


    # %%
    LDAvis_prepared


    # %%
    # encodedDatasetArgs = {'splitter': splitter,
    #                               'tokenizerConfig': tokenizer_config,
    #                               'cleaning_function': text_pre_cleaning_function}

    # logger.info('Cleaned!')
    # for train_dataset, val_dataset in model.pipeLine.getEncodedDataset(**encodedDatasetArgs):
    #             pass

if __name__ == "__main__":
    main()

# %%
