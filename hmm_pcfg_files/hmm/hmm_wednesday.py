from hmm_pcfg_files.tools import read_from_file, score_tag, score_tag_without_emits
import numpy as np
import pandas as pd
import csv

hmm_emits_pd = pd.read_csv("hmm_pcfg_files/hmm_emits", sep="\t", quoting=csv.QUOTE_NONE, header=None)
distinct_tags = np.unique(hmm_emits_pd[0])
distinct_words = np.unique(hmm_emits_pd[1])
hmm_emits_pd = hmm_emits_pd.set_index([0, 1])

hmm_trans_pd = pd.read_csv("hmm_pcfg_files/hmm_trans", sep="\t", quoting=csv.QUOTE_NONE, header=None)
distinct_tags_and_sentence_boundary = np.unique(hmm_trans_pd[0])

assert len(distinct_tags) == len(distinct_tags_and_sentence_boundary) - 1
# Only the 'sentence_boundary' tag is missing

hmm_trans_pd = hmm_trans_pd.set_index([0, 1])

test_sent = read_from_file("hmm_pcfg_files/dev_sents")
test_sent = [sentence[0].split(" ") for sentence in test_sent]
hmm_trans_pd = hmm_trans_pd.apply(lambda x: np.exp(x))
hmm_emits_pd = hmm_emits_pd.apply(lambda x: np.exp(x))

def tag_viterbi(sentence):
    viterbi = []  # [first_viterbi, score_1, score_2, ...] array of dictionaries
    back_pointer = []  # [first_back_pointer, back_pointer_1, back_pointer_2, ...] array of dictionaries

    first_viterbi = {}
    first_back_pointer = {}
    for tag in distinct_tags:
        # don't record anything for the START tag
        # if tag == "START": continue
        if not (tag, sentence[0]) in hmm_emits_pd.index:
            first_viterbi[tag] = 0
        else:
            first_viterbi[tag] = hmm_trans_pd.ix["sentence_boundary", tag].values[0] * \
                                 hmm_emits_pd.ix[tag, sentence[0]].values[0]
        first_back_pointer[tag] = "sentence_boundary"

    viterbi.append(first_viterbi)
    back_pointer.append(first_back_pointer)

    for word_index in range(1, len(sentence)):
        this_viterbi = {}
        this_back_pointer = {}
        prev_viterbi = viterbi[-1]

        for tag in distinct_tags:
            best_previous = max(prev_viterbi.keys(),
                                key=lambda previous_tag: score_tag(tag,
                                                                   previous_tag,
                                                                   sentence,
                                                                   word_index,
                                                                   prev_viterbi,
                                                                   hmm_trans_pd,
                                                                   hmm_emits_pd,
                                                                   )
                                )

            if not (((tag, sentence[word_index]) in hmm_emits_pd.index) and (
                        (best_previous, tag) in hmm_trans_pd.index)):
                this_viterbi[tag] = 0
            else:
                this_viterbi[tag] = prev_viterbi[best_previous] * \
                                    hmm_trans_pd.ix[best_previous, tag].values[0] * \
                                    hmm_emits_pd.ix[tag, sentence[word_index]].values[0]

            this_back_pointer[tag] = best_previous

        viterbi.append(this_viterbi)
        back_pointer.append(this_back_pointer)

    # done with all words in the sentence.
    # now find the probability of each tag to have "END" as the next tag,
    # and use that to find the overall best sequence
    prev_viterbi = viterbi[-1]

    best_previous = max(prev_viterbi.keys(),
                        key=lambda previous_tag: score_tag_without_emits("sentence_boundary",
                                                                         previous_tag,
                                                                         prev_viterbi,
                                                                         hmm_trans_pd))

    best_tag_sequence = ["sentence_boundary", best_previous]
    back_pointer.reverse()

    current_best_tag = best_previous
    for bp in back_pointer:
        best_tag_sequence.append(bp[current_best_tag])

    best_tag_sequence.reverse()
    return " ".join(best_tag_sequence[1:-1])


