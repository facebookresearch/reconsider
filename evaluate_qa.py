# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np
from collections import defaultdict

def get_qa_predictions(logger, all_features, outputs, max_answer_length, n_paragraphs,
                       use_regex=False, freebase_entities=None, write_prediction=False, unidecode=False, wikiTitles=None):
    all_results = []
    for (example_index, feature) in enumerate(all_features):
        switch_logits_list = outputs[example_index]
        assert type(switch_logits_list[0])==float

        if write_prediction:
            softmax_switch_logits_list = _compute_softmax(switch_logits_list)

        sorted_logits = sorted(enumerate(switch_logits_list),
                               key=lambda x: -x[1])
        best_psg_idx, best_score = sorted_logits[0]

        all_results.append({'id':feature['id'],
                            'prediction': best_psg_idx,
                            'passage': " ".join(feature['positive_doc_tokens'][best_psg_idx]),
                            'question': feature['question'],
                            'groundtruth': feature['answers']})

    f1s, ems = [], []

    for pred_dict in all_results:
        groundtruth = pred_dict['groundtruth']
        ems.append(int(pred_dict['prediction'] in groundtruth))
    em = np.mean(ems)
    logger.info("n=%d\tF1 %.2f\tEM %.2f"%(0, 0*100, em*100))

    return {'em': em, 'f1': 0}, all_results

def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    if type(scores[0])==tuple:
        scores = [s[1] for s in scores]

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs
