# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import shutil
import sys
from hotpot_evaluate_v1 import normalize_answer, regex_match_score
from pytorch_transformers import BasicTokenizer
import random
import json

def load_jsonl(f):
  js = []
  for line in open(f):
    js.append(json.loads(line))
  return js

parser = argparse.ArgumentParser(description='train_qa.py')

parser.add_argument('--answer_json', default = '',
                    help='which model to run')
parser.add_argument('--orig_json', default = 'nq',
                    help='which model to run')
parser.add_argument('--out_json', default = '/checkpoint/sewonmin/data/nq-dpr/nq-train-multi.json',
                    help='which model to run')
parser.add_argument('--dev', action='store_true',
                    help='which model to run')
parser.add_argument('--test_M', type=int, default=5,
                    help='which model to run')
parser.add_argument('--train_M', type=int, default=100,
                    help='which model to run')
args = parser.parse_args()

out = open(args.out_json, 'w')

tokenize = BasicTokenizer().tokenize

answer_json = load_jsonl(args.answer_json)
orig_json = load_jsonl(args.orig_json)

all_negative = 0
all_positive = 0
pred_doesnt_matter = 0
accuracy = 0
max_answers = args.test_M if args.dev else args.train_M
count = 0

for (answers, orig) in zip(answer_json, orig_json):
  count += 1
  if count % 1000 == 0:
    print(count)
  groundtruth = [normalize_answer(gt) for gt in orig['answers']] if not 'trec' in args.orig_json else orig['answers']
  out_json = { "id": "1", "gt_title": None, "answers": [], "question": '', "positives": [], "negatives": [] }
  at_least_1_correct = False

  if 'questions' not in orig:
    orig['questions'] = [orig['question']]

  for ans_idx, ans in enumerate(answers): # 5-25 answers
    if ans_idx >= max_answers:
      break
    passage_index = ans['passage_index']

    question_p = tokenize(orig['question'])

    answer_begin_index = ans['passage'].find('<answer>')
    answer_end_index = ans['passage'].find('</answer>')
    marked_passage = tokenize(ans['passage'][:answer_begin_index]) + ['[unused0]'] + tokenize(ans['passage'][(answer_begin_index + 8):(answer_end_index)]) + ['[unused1]'] + tokenize(ans['passage'][answer_end_index + 10:])

    final_passage = question_p + ['[SEP]'] +  tokenize(ans['title']) + ['[SEP]'] + marked_passage

    if not 'trec' in args.orig_json:
      correct = normalize_answer(ans['text']) in groundtruth 
    else:
      correct = max([regex_match_score(ans['text'], gt) for gt in groundtruth]) # This returns a bool

    if correct:
      at_least_1_correct =  True

    if correct or args.dev:
      out_json['positives'].append(["", final_passage, []])
    else:
      out_json['negatives'].append(["", final_passage, []])

    if correct:
      out_json['answers'].append(ans_idx)

  if len(out_json['negatives']) == 0:
    all_positive += 1

  if len(out_json['positives']) == 0:
    all_negative += 1

  if not args.dev and (len(out_json['negatives']) == 0 or len(out_json['positives']) == 0):
    continue

  if args.dev and len(out_json['positives']) == len(out_json['answers']):
    pred_doesnt_matter += 1

  if at_least_1_correct:
    accuracy += 1

  out.write(json.dumps(out_json) + '\n')

out.close()
print(all_positive)
print(all_negative)
print(pred_doesnt_matter)
print(accuracy)
