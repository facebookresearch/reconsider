# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import pickle as pkl
import collections
from tqdm import tqdm
from multiprocessing import Pool
from itertools import product

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from DataLoader import MyQADataLoader

def get_dataloader(logger, args, input_file, is_training, num_epochs, tokenizer):

    feature_save_path = input_file.replace('.json', '.pkl')
    feature_save_path = feature_save_path.replace('.pkl', '-pl{}.{}.{}.{}.pkl'.format(args.max_passage_length, args.max_question_length, args.pad_question, args.max_negatives))
    if args.bert_name != "bert-base-uncased":
        feature_save_path = feature_save_path.replace('.pkl', '-{}.pkl'.format(args.bert_name))

    if os.path.exists(feature_save_path) and not args.no_pkl: # and not is_training:
        logger.info("Loading saved features from {}".format(feature_save_path))
        with open(feature_save_path, 'rb') as f:
            features = pkl.load(f)
        logger.info("Loaded saved features from {}".format(feature_save_path))
    else:
        logger.info("{} not found. Computing features".format(feature_save_path))
        read_examples = read_qa_examples
        features = read_examples(
            logger=logger, args=args, input_file=input_file,
            tokenizer=tokenizer, is_training=is_training)
        if not args.debug:
            logger.info("Saving features to: {}".format(feature_save_path))
            with open(feature_save_path, 'wb') as f:
                pkl.dump(features, f)
    features = features['features']
    n_features = len(features)
    num_train_steps = int(len(features) / args.train_batch_size * num_epochs)

    logger.info("  Num features = %d", n_features)
    logger.info("  Batch size = %d", args.train_batch_size if is_training else args.predict_batch_size)
    if is_training:
        logger.info("  Num steps = %d", num_train_steps)

    dataloader = MyQADataLoader(args=args, features=features, is_training=is_training,
                                batch_size=args.train_batch_size if is_training else args.predict_batch_size)
    if is_training:
        return features, dataloader, num_train_steps
    return features, dataloader

def process_qa_file_entry(entry, is_training, tokenizer, args):


  positive_input_ids = []
  positive_input_mask = []
  positive_start_positions = []
  positive_end_positions = []
  positive_answer_mask = []
  positive_doc_tokens = [p[1] for p in entry['positives']]
  positive_tokens = []
  positive_tok_to_orig_map = []
  negative_input_ids = []
  negative_input_mask = []
  num_truncated = 0
  num_q_tokens = 0
  assert args.debug or (is_training and len(entry['positives'])>0) \
      or (not is_training and len(entry['positives'])>=0 and len(entry['negatives'])==0)
  if (not is_training) and len(entry['positives'])==len(entry['negatives'])==0:
      print ("Dev example with no retrieval found")
      entry['positives'] = [["dummy", "dummy", []]]

  # entry is a json line from the input file. Contains question, answers, positives and negatives
  for idx, passage in enumerate(entry['positives'][:100]):
      input_ids, input_mask, tokens, tok_to_orig_map, start_positions, end_positions, answer_mask, truncated, q_tokens = \
          convert_qa_feature(tokenizer, entry['question'], passage,
                             max_length=args.max_passage_length,
                             max_n_answers=args.max_n_answers,
                             compute_span=is_training, similar_answers=entry['similar_answers'] if args.similar_answers else None, args=args)
      num_truncated += truncated
      if idx == 0: # its the same question with multiple passages
        num_q_tokens += q_tokens
      positive_input_ids.append(input_ids)
      positive_input_mask.append(input_mask)
      positive_tokens.append(tokens)
      positive_tok_to_orig_map.append(tok_to_orig_map)
      positive_start_positions.append(start_positions)
      positive_end_positions.append(end_positions)
      positive_answer_mask.append(answer_mask)
  for i, passage in enumerate(entry['negatives']):
      input_ids, input_mask, _, _, _, _, _, truncated, q_tokens = convert_qa_feature(tokenizer,
                                              entry['question'],
                                              passage,
                                              max_length=args.max_passage_length,
                                              max_n_answers=args.max_n_answers,
                                              compute_span=False, similar_answers=entry['similar_answers'] if args.similar_answers else None, args=args)
      num_truncated += truncated
      negative_input_ids.append(input_ids)
      negative_input_mask.append(input_mask)
  return {
          'id': entry['id'],
          'positive_input_ids': positive_input_ids,
          'positive_input_mask': positive_input_mask,
          'positive_tokens': positive_tokens,
          'positive_doc_tokens': positive_doc_tokens,
          'positive_tok_to_orig_map': positive_tok_to_orig_map,
          'positive_start_positions': positive_start_positions,
          'positive_end_positions': positive_end_positions,
          'positive_answer_mask': positive_answer_mask,
          'negative_input_ids': negative_input_ids,
          'negative_input_mask': negative_input_mask,
          'question': entry['question'],
          'gt_title': entry['gt_title'],
          'answers': entry['answers'],
          'truncated': num_truncated,
          'q_tokens': num_q_tokens
      }

def read_qa_examples(logger, args, input_file, is_training, tokenizer, features=None):
  input_data = []
  logger.info("Loading {}".format(input_file))
  with open(input_file, "r") as f:
      for line in f:
          js = json.loads(line)
          js['negatives'] = js['negatives'][:args.max_negatives]
          js['positives'] = js['positives'][:args.max_positives]
          input_data.append(js)
          if args.debug and len(input_data)==20:
              break
  logger.info("File Loaded. Now computing features.")
  return read_qa_examples_from_json(logger, args, input_data, is_training, tokenizer, features)


def read_qa_examples_from_json(logger, args, input_data, is_training, tokenizer, features):
    features = []

    if args.debug:
      for ind in input_data:
        features.append(process_qa_file_entry(ind, is_training, tokenizer, args))
    else:
      with Pool(processes=args.threads) as pool:
        features = pool.starmap(process_qa_file_entry, product(input_data, [is_training], [tokenizer], [args]))


    total_truncated = 0
    total_q_tokens = 0
    total_num = 0
    for f in features:
      total_q_tokens += f['q_tokens']
      total_truncated += f['truncated']
      total_num += len(f['negative_input_ids']) + len(f['positive_input_ids'])
    logger.info('Total truncated: {}/{}'.format(total_truncated, total_num))
    logger.info('Average q tokens: {}/{}'.format(total_q_tokens, len(features)))


    return {'features': features}


def convert_qa_feature(tokenizer, question, passage, max_length,
                       max_n_answers, compute_span, similar_answers, args):

    # passage is a single positive or negative passage. passage[0] is the title. passage[1] is the passage tokens.
    # passage[2] is a list of dicts of the form {text: "answer text", "answer_start": start_pos, "word_start": word_start_pos, "word_end": word_end_pos}
    # There is one dict for each occurrence of the answer
    # answer_start is the character index of the answer start
    # word_start is the word index of the start word

    question_tokens = tokenizer.tokenize(question)
    if args.pad_question:
      question_tokens = question_tokens + ['[PAD]'] * max(0, args.max_question_length - len(question_tokens))

    title_tokens = tokenizer.tokenize(passage[0])

    passage_tokens = []
    tok_to_orig_index = []
    orig_to_tok_index = []

    # Here, we are tokenizing the tokens using BertTokenizer. The tokens were originally created
    # using BasicTokenizer. We tokenize twice since we have to map subtokens back to original tokens
    # to accurately compute the answer span
    for (i, token) in enumerate(passage[1]):
        orig_to_tok_index.append(len(passage_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            passage_tokens.append(sub_token)

    # orig_to_tok_index is of size len(orig tokens). Each positions tells us the start of the subtoken index 
    # that the subtoken maps to

    # tok_to_orig is of size len(subtokens). Each position tells us the index of the original token
    if similar_answers:
      for ans in similar_answers:
        passage_tokens += ['[SEP]'] + tokenizer.tokenize(ans)

    tokens = []
    tokens.append("[CLS]")
    for token in question_tokens:
        tokens.append(token)
    if len(question_tokens) > 0:
      tokens.append("[SEP]")
    for token in title_tokens:
        tokens.append(token)
    if len(title_tokens) > 0:
      tokens.append("[SEP]")
    for token in passage_tokens:
        tokens.append(token)

    truncated = 0
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
        truncated = 1
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_length:
        input_ids.append(0)
        input_mask.append(0)
    assert len(input_ids) == max_length
    assert len(input_mask) == max_length

    # input_mask, is which positions are proper input tokens
    # and which ones are padding. Padding is added above i.e. the 0s

    offset = len(question_tokens) + len(title_tokens) + 3
    token_to_orig_map = {token+offset:orig for token, orig in enumerate(tok_to_orig_index)}

    # add offset for the question and the two [SEP] and [CLS] tokens and 
    # make a dict for easy access

    if compute_span: # used during training
        start_positions, end_positions = [], []

        for answer in passage[2]: # passage[2] is a list of dicts, one dict for each answer. See above for its contents
            tok_start_position = offset + orig_to_tok_index[answer['word_start']] # get subtoken index of answer word start in #format

            # now deal with end
            if len(orig_to_tok_index)==answer['word_end']+1:
                tok_end_position = offset + orig_to_tok_index[answer['word_end']] # set it to start of the word?
            else:
                tok_end_position = offset + orig_to_tok_index[answer['word_end']+1]-1 # Next token start - 1

            if tok_end_position > max_length: # We cant use this. Continue checking the next answer
                continue

            start_positions.append(tok_start_position) # Multiple start positions for this passage
            end_positions.append(tok_end_position) # Multiple corresponding end positions
        if len(start_positions) > max_n_answers: # truncate to maximum answers
            start_positions = start_positions[:max_n_answers]
            end_positions = end_positions[:max_n_answers]

        answer_mask = [1 for _ in range(len(start_positions))]
        # We need to have max_n_answers values. answer_mask denotes which of those are valid
        for _ in range(max_n_answers-len(start_positions)): # Pad the start_positions, end_positions, and answer_mask arrays to reach max_n_answers
            start_positions.append(0)
            end_positions.append(0)
            answer_mask.append(0)
    else:
        # for evaluation
        start_positions, end_positions, answer_mask = None, None, None
    return input_ids, input_mask, \
        tokens, token_to_orig_map, \
        start_positions, end_positions, answer_mask, truncated, len(question_tokens)


