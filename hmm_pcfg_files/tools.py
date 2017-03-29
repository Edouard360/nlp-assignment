import csv


def read_from_file(path):
    with open(path, "r") as f:
        reader = csv.reader(f)
        return list(reader)

def get_value(df,index_1,index_2):
    if (index_1,index_2) not in df.index:
        return 0
    else:
        return df.ix[index_1,index_2].values[0]

def score_tag(tag, previous_tag, sentence, word_index, prev_viterbi, hmm_trans_pd, hmm_emits_pd):
    """
    :param tag: the tag considered in this iteration
    :param previous_tag: the tag of the previous iteration, we want to find its optimal value to maximize our score
    :param hmm_trans_pd: transition matrix
    :param hmm_emits_pd: emission matrix
    :param prev_viterbi: previous score
    :return: return the score of the tag, given the previous tag
    """
    if not (((tag, sentence[word_index]) in hmm_emits_pd.index) and
                ((previous_tag, tag) in hmm_trans_pd.index)):
        score = 0
    else:
        score = prev_viterbi[previous_tag] * \
                hmm_trans_pd.ix[previous_tag, tag].values[0] * \
                hmm_emits_pd.ix[tag, sentence[word_index]].values[0]
    return score

def score_tag_without_emits(tag, previous_tag, prev_viterbi, hmm_trans_pd):
    if not ((previous_tag, tag) in hmm_trans_pd.index):
        score = 0
    else:
        score = prev_viterbi[previous_tag] * \
                hmm_trans_pd.ix[previous_tag, tag].values[0]
    return score
