import nltk
import csv
import pandas as pd
import numpy as np
from math import exp

hmm_emits_pd = pd.read_csv("hmm_emits", sep="\t", quoting=csv.QUOTE_NONE, header=None)
distinct_words = set(np.unique(hmm_emits_pd[1]))
distinct_tags = set(np.unique(hmm_emits_pd[0]))

arabic_pcfg = open("pcfg").read()
arabic_pcfg = arabic_pcfg.splitlines()

for i, rule in enumerate(arabic_pcfg):
    rule = rule.replace("PRP$", "PRP2")
    split_rule = rule.split("\t")
    if split_rule[1] in distinct_words:
        # we add " " for lexical production (see lab7)
        if split_rule[1] == '"':
            split_rule[1] = "'"+split_rule[1]+"'"
        else:
            split_rule[1] = '"' + split_rule[1] + '"'

    # min for the float is 7.981037055632809e-06 and scientific notation not allowed...
    split_rule[2] = "%.10f" % exp(float(split_rule[2])) #

    float_prob = exp(float(split_rule[2]))
    arabic_pcfg[i] = split_rule[0]+" -> "+split_rule[1]+" ["+split_rule[2]+"]" # update arabic_pcfg inplace


arabic_pcfg = "\n".join(arabic_pcfg)

nltk_pcfg = nltk.PCFG.fromstring(arabic_pcfg)



#arabic_pcfg_df = pd.DataFrame(data=arabic_pcfg)