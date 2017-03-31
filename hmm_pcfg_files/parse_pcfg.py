import nltk
import csv
import pandas as pd
import numpy as np
from math import exp
from nltk.app import srparser, rdparser
from hmm_pcfg_files.tools import read_from_file

hmm_emits_pd = pd.read_csv("hmm_emits", sep="\t", quoting=csv.QUOTE_NONE, header=None)
distinct_words = set(np.unique(hmm_emits_pd[1]))
distinct_tags = set(np.unique(hmm_emits_pd[0]))

arabic_pcfg = open("pcfg").read()
arabic_pcfg = arabic_pcfg.splitlines()
distinct_words_from_pcfg = set()
# Correcting the source file
for i, rule in enumerate(arabic_pcfg):
    rule = rule.replace("PRP$", "PRP2")
    split_rule = rule.split("\t")
    if split_rule[1] in distinct_words:
        # we add " " for lexical production (see lab7)
        distinct_words_from_pcfg.add(split_rule[1])
        if split_rule[1] == '"':
            split_rule[1] = "'" + split_rule[1] + "'"
        else:
            split_rule[1] = '"' + split_rule[1] + '"'

    # min for the float is 7.981037055632809e-06 and scientific notation not allowed...
    split_rule[2] = "%.10f" % exp(float(split_rule[2]))  #

    float_prob = exp(float(split_rule[2]))
    arabic_pcfg[i] = split_rule[0] + " -> " + split_rule[1] + " [" + split_rule[2] + "]"  # update arabic_pcfg inplace

distinct_words.symmetric_difference(distinct_words_from_pcfg)
arabic_pcfg = "\n".join(arabic_pcfg)
grammar = nltk.PCFG.fromstring(arabic_pcfg)

test_sent = read_from_file("dev_sents")
test_sent = [test[0].split(" ") for test in test_sent]

from nltk import parse, pchart

parser = parse.RecursiveDescentParser(grammar, trace=2)

# for p in parser.parse(test_sent[0]):
#     print(p)

pchart_parser = pchart.RandomChartParser(grammar)

for p in pchart_parser.parse(test_sent[0]):
    print(p)

#
# list(parser.parse(test))
# output = " ".join([str(p) for p in parser.parse(test)])
#
# # Not working aon
# with open('parses/candidate_parses', 'w') as f:
#     for sentence in test_sent:
#         output = " ".join([p for p in parser.parse(test_sent)])
#         f.write(output + "\n")
#     f.close()
