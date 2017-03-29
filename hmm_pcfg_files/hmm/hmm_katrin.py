from hmm_pcfg_files.tools import read_from_file, get_value
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

test_sent = read_from_file("hmm_pcfg_files/dev_sents")
test_sent = [test[0].split(" ") for test in test_sent]

def tag_viterbi(sentence):
    viterbi = []
    backpointer = []

    first_viterbi = {}
    first_backpointer = {}

    for tag in distinct_tags:
        first_viterbi[tag] = get_value(hmm_trans_pd,"sentence_boundary", tag)*get_value(hmm_emits_pd,tag, sentence[0])
        first_backpointer[tag] = "sentence_boundary"

    viterbi.append(first_viterbi)
    backpointer.append(first_backpointer)

    for wordindex in range(1, len(sentence)):
        this_viterbi = {}
        this_backpointer = {}
        prev_viterbi = viterbi[-1]

        for tag in distinct_tags:
            best_previous = max(prev_viterbi.keys(),
                                key=lambda prevtag: \
                                    prev_viterbi[prevtag] * get_value(hmm_trans_pd,prevtag,tag) * get_value(hmm_emits_pd,tag,sentence[wordindex]))

            this_viterbi[tag] = prev_viterbi[best_previous] * \
                                get_value(hmm_trans_pd, best_previous, tag) * \
                                get_value(hmm_emits_pd, tag, sentence[wordindex])
            this_backpointer[tag] = best_previous

        viterbi.append(this_viterbi)
        backpointer.append(this_backpointer)

    # done with all words in the sentence.
    # now find the probability of each tag
    # to have "END" as the next tag,
    # and use that to find the overall best sequence
    prev_viterbi = viterbi[-1]
    best_previous = max(prev_viterbi.keys(),
                        key=lambda prevtag: prev_viterbi[prevtag] * get_value(hmm_trans_pd,prevtag,"sentence_boundary"))

    prob_tagsequence = prev_viterbi[best_previous] * get_value(hmm_trans_pd,best_previous,"sentence_boundary")

    # best tagsequence: we store this in reverse for now, will invert later
    best_tagsequence = ["sentence_boundary", best_previous]
    # invert the list of backpointers
    backpointer.reverse()

    current_best_tag = best_previous
    for bp in backpointer:
        best_tagsequence.append(bp[current_best_tag])
        current_best_tag = bp[current_best_tag]

    best_tagsequence.reverse()
    return " ".join(best_tagsequence[1:-1])

with open('hmm_pcfg_files/postags/postags_katrin', 'w') as f:
    for sentence in test_sent:
        output = tag_viterbi(sentence)
        f.write(output+"\n")
    f.close()