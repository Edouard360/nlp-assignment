from hmm_pcfg_files.tools import read_from_file
import numpy as np
import pandas as pd
import csv
import sys, time
from nltk import tokenize
from nltk.parse import ViterbiParser
from nltk.grammar import toy_pcfg1, toy_pcfg2
from nltk import grammar, parse
import nltk
from nltk.grammar import ContextFreeGrammar, Nonterminal

test_sent = read_from_file("hmm_pcfg_files/dev_sents")
test_sent = [test[0].split(" ") for test in test_sent]

cp = parse.load_parser('hmm_pcfg_files/pcfg', trace=1, format='pcfg')


with open('hmm_pcfg_files/parses/candidate-parses', 'w') as f:
    for sentence in test_sent:

        output = " ".join([tag[1] for tag in tag_sequence])
        f.write(output+"\n")
    f.close()
