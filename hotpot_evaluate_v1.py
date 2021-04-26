# This file is borrowed from https://github.com/hotpotqa/hotpot
# Licensed under Apache License 2.0

import sys
import re
import string
from collections import Counter
import pickle
from pytorch_transformers import BasicTokenizer


def largest_subspan_in_wiki(bt, setWikiTitles, ans, min_len=2):
    ret = []
    lst_ans_tok = bt.tokenize(ans)
    if len(lst_ans_tok) < min_len:
        return ret

    stop = False
    for l in range(len(lst_ans_tok),min_len-1,-1):
        for s in range(len(lst_ans_tok)-l+1):
            sub_span = ' '.join(lst_ans_tok[s:s+l])
            blMatch = sub_span in setWikiTitles
            if blMatch:
                stop = True
                ret.append(sub_span)
                #print (l, s, sub_span, sub_span in setWikiTitles)
        if stop:
            break
    return ret

def wikiSpan(s, wikiTitles):
  if wikiTitles:
    bt = BasicTokenizer()
    blIgnore = False
    for tok in bt.tokenize(s):
      if tok.isnumeric():
        blIgnore = True

    if not blIgnore:
      lstSpan = largest_subspan_in_wiki(bt, wikiTitles, s)
      if len(lstSpan) > 0:
        return lstSpan[0]
  return s

def normalize_answer(s, wikiTitles=None):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return wikiSpan(white_space_fix(remove_articles(remove_punc(lower(s)))), wikiTitles)


def f1_score(prediction, ground_truth, normalized=False):
    if normalized:
        normalized_prediction = prediction
        normalized_ground_truth = ground_truth
    else:
        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth, wikiTitles=None):
    return (normalize_answer(prediction, wikiTitles) == normalize_answer(ground_truth, wikiTitles))


def regex_match_score(prediction, pattern):
    """Check if the prediction matches the given regular expression."""
    try:
        compiled = re.compile(
            pattern,
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE
        )
    except BaseException:
        print('Regular expression failed to compile: %s' % pattern)
        return False
    return compiled.match(prediction) is not None

def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    metrics['em'] += em
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    return em, prec, recall

def update_sp(metrics, prediction, gold):
    cur_sp_pred = set(map(tuple, prediction))
    gold_sp_pred = set(map(tuple, gold))
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    metrics['sp_em'] += em
    metrics['sp_f1'] += f1
    metrics['sp_prec'] += prec
    metrics['sp_recall'] += recall
    return em, prec, recall

