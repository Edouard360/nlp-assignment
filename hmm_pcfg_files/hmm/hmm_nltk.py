from nltk.probability import DictionaryConditionalProbDist, DictionaryProbDist
from nltk.tag import HiddenMarkovModelTagger

from hmm_pcfg_files.tools import read_from_file
import numpy as np
import pandas as pd
import csv

hmm_emits_pd = pd.read_csv("hmm_pcfg_files/hmm_emits", sep="\t", quoting=csv.QUOTE_NONE, header=None)
distinct_tags = np.unique(hmm_emits_pd[0])
distinct_words = np.unique(hmm_emits_pd[1])
hmm_trans_pd = pd.read_csv("hmm_pcfg_files/hmm_trans", sep="\t", quoting=csv.QUOTE_NONE, header=None)

hmm_emits_pd = hmm_emits_pd.set_index([0, 1])
hmm_trans_pd = hmm_trans_pd.set_index([0, 1])
hmm_trans_pd = hmm_trans_pd.apply(lambda x: np.exp(x))
hmm_emits_pd = hmm_emits_pd.apply(lambda x: np.exp(x))

tag_dict_tag = dict()
for tag in distinct_tags:
    tag_dict = dict(zip(hmm_trans_pd.ix[tag].index, hmm_trans_pd.ix[tag].values.ravel()))
    #missing_to_dict = list(set(distinct_tags).difference(tag_dict.keys()))
    #tag_dict.update(zip(missing_to_dict,np.zeros(len(missing_to_dict))))
    tag_dict_tag[tag] = DictionaryProbDist(tag_dict)

transition = DictionaryConditionalProbDist(tag_dict_tag)

tag_dict_word = dict()
for tag in distinct_tags:
    tag_dict = dict(zip(hmm_emits_pd.ix[tag].index, hmm_emits_pd.ix[tag].values.ravel()))
    #missing_to_dict = list(set(distinct_tags).difference(tag_dict_word.keys()))
    #tag_dict_word.update(zip(missing_to_dict,np.zeros(len(missing_to_dict))))
    tag_dict_word[tag] = DictionaryProbDist(tag_dict)

emission = DictionaryConditionalProbDist(tag_dict_word)

def get_value(df,index_1,index_2):
    if (index_1,index_2) not in df.index:
        return 0
    else:
        return df.ix[index_1,index_2].values[0]

symbols = distinct_words
states = distinct_tags

test_sent = read_from_file("hmm_pcfg_files/dev_sents")
test_sent = [test[0].split(" ") for test in test_sent]

with open('hmm_pcfg_files/postags/postags_nltk', 'w') as f:
    for sentence in test_sent:
        prior = {}
        for tag in distinct_tags:
            value = get_value(hmm_trans_pd, "sentence_boundary", tag) * get_value(hmm_emits_pd, tag, sentence[0])
            if value != 0: prior[tag] = value
        prior = DictionaryProbDist(prior)
        tagger = HiddenMarkovModelTagger(symbols, states, transition, emission, prior)
        tag_sequence = tagger.tag(sentence)
        output = " ".join([tag[1] for tag in tag_sequence])
        f.write(output+"\n")
    f.close()
