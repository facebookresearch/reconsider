# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import collections
import logging
import json
import math
import os
import random
import six
from tqdm import tqdm
import time
import pickle

import numpy as np
import heapq
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch_transformers import (WEIGHTS_NAME,
                                  BertConfig, BertTokenizer, RobertaConfig, RobertaTokenizer)
from modeling import QAModel
from pytorch_transformers import AdamW, WarmupLinearSchedule

from prepro import get_dataloader
from evaluate_qa import get_qa_predictions


def main():
    parser = argparse.ArgumentParser()

    ## Basic parameters
    parser.add_argument("--train_file", default=None, type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default=None, type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_predict", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--wikiNormalize", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--similar_answers", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--pad_question", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--ensemble", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument('--question_pool_fn', type=str, default='mean')
    parser.add_argument("--no_softmax", action='store_true',
                        help="Whether to run eval on the dev set.")

    ## Pretrained Model parameters
    parser.add_argument('--bert_name', type=str, default='bert-base-uncased')
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--checkpoint", type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).", \
                        default=None)

    # Preprocessing-related parameters
    parser.add_argument('--max_passage_length', type=int, default=200)
    parser.add_argument('--max_negatives', type=int, default=20000)
    parser.add_argument('--max_positives', type=int, default=20000)
    parser.add_argument('--max_question_length', type=int, default=32)
    parser.add_argument('--train_M', type=int, default=2)
    parser.add_argument('--test_M', type=int, default=50)
    parser.add_argument("--threads", type=int, default=10, help="Number of threads for data loading")

    # Training-related parameters
    # parser.add_argument("--num_train", default=5, type=int,
                        # help="Batch size per GPU/CPU for training.")
    parser.add_argument("--print_every", default=50, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--train_batch_size", default=40, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--predict_batch_size", default=400, type=int,
                        help="Batch size per GPU/CPU for evaluation.")

    parser.add_argument("--bert_learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--learning_rate", default=0.001, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=100000.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--wait_step', type=int, default=10)
    parser.add_argument('--train_splits', type=int, default=1)
    parser.add_argument("--scheduler", default="warmup", type=str,
                        help="Learning method for weak supervision setting")

    ## Latent variable learning
    parser.add_argument("--max_n_answers", default=10, type=int)
    parser.add_argument("--loss_type", default="mml", type=str,
                        help="Learning method for weak supervision setting")
    parser.add_argument('--tau', type=float, default=12000.0, help="Needed for Hard EM")

    ## Evaluation-related parameters
    parser.add_argument("--n_best_size", default=1, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--max_answer_length", default=10, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument('--eval_period', type=int, default=400,
                        help="Evaluate & save model")
    parser.add_argument('--prefix', type=str, default='',
                        help="Prefix for saving predictions")
    parser.add_argument('--n_paragraphs', type=str, default=None)

    ## Other parameters
    parser.add_argument('--debug', action='store_true',
                        help="Use a subset of data for debugging")
    parser.add_argument('--no_pkl', action='store_true',
                        help="Use a subset of data for debugging")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    n_gpu = torch.cuda.device_count()
    devices = [torch.device("cuda", i) for i in range(n_gpu)]

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    ##### Start writing logs

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=[logging.FileHandler(os.path.join(args.output_dir, "log.txt")),
                              logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)
    logger.info(args.output_dir)
    logger.info('Cuda version: {}'.format(torch.version.cuda))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError("If `do_train` is True, then `train_file` must be specified.")
        if not args.predict_file:
            raise ValueError("If `do_train` is True, then `predict_file` must be specified.")

    if args.do_predict:
        if not args.predict_file:
            raise ValueError("If `do_predict` is True, then `predict_file` must be specified.")

    if args.bert_name.startswith('bert'):
        Config = BertConfig
        Tokenizer = BertTokenizer
    elif args.bert_name.startswith('roberta'):
        Config = RobertaConfig
        Tokenizer = RobertaTokenizer

    config = Config.from_pretrained(args.bert_name)
    tokenizer = Tokenizer.from_pretrained(args.bert_name)
    special_tokens_dict = {'additional_special_tokens': ['[unused0]', '[unused1]', '[unused2]']}
    tokenizer.add_special_tokens(special_tokens_dict)

    model = QAModel(config, devices, bert_name=args.bert_name,
                    loss_type=args.loss_type, tau=args.tau)
    main_metric = "em"
    second_metric = "f1"
    features, dataloader = get_dataloader(
                logger=logger, args=args,
                input_file=args.predict_file,
                is_training=False,
                num_epochs=1,
                tokenizer=tokenizer)
    if args.do_train:
        # We need to do this here even when loading splits, beacuse we need num_train_steps
        train_features, train_dataloader, num_train_steps = get_dataloader(
                logger=logger, args=args,
                input_file=args.train_file if args.train_splits == 1 else args.train_file.replace('-train', '-train0'),
                is_training=True,
                num_epochs=args.num_train_epochs,
                tokenizer=tokenizer)

    if 'webq' in args.predict_file:
        from hotpot_evaluate_v1 import normalize_answer
        freebase_entities = set()
        with open('freebase-entities.txt', 'r') as f:
            for line in f:
                freebase_entities.add(normalize_answer(line.strip()))
        logger.info("Loaded {} freebase entities for evaluation".format(len(freebase_entities)))
    else:
        freebase_entities = None

    if args.checkpoint is not None or args.do_predict:
        if args.checkpoint is not None:
            checkpoint = args.checkpoint
        else:
            checkpoint = os.path.join(args.output_dir, 'best-model.pt')
        logger.info("Loading from {}".format(checkpoint))
        def filter_func(key):
            if key.startswith('module.'):
                return key[7:]
            return key
        model = model.from_pretrained(model, checkpoint, filter_func, config=config)
    else:
        model.bert = model.bert.from_pretrained(None, args.bert_name, None, config=config)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.to(torch.device("cuda"))

    if args.do_train:
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'bert' in n],
         'weight_decay': args.weight_decay, 'lr': args.bert_learning_rate},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'bert' in n],
         'weight_decay': 0.0, 'lr': args.bert_learning_rate},
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and not 'bert' in n], 'lr': args.learning_rate, 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and not 'bert' in n], 'lr': args.learning_rate, 'weight_decay': 0.0}
    ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        if args.scheduler == "plateau":
          scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=0,
                        verbose=True)
        elif args.scheduler == "warmup":
          scheduler = WarmupLinearSchedule(optimizer,
                                          warmup_steps=args.warmup_steps,
                                          t_total=num_train_steps * args.train_splits)

        logger.info("Training start with {} gpus".format(n_gpu))
        global_step = 0
        best_accuracy = (-1, -1)
        wait_step = 0
        model.train()
        global_step = 0
        stop_training = False
        train_losses = []
        eval_period_updated = False
        BS = args.train_batch_size * args.train_M

        for epoch in range(int(args.num_train_epochs)):
            if args.train_splits > 1:
              _ , train_dataloader, _ = get_dataloader(
                      logger=logger, args=args, \
                      input_file=args.train_file.replace('-train', '-train{}'.format(epoch % args.train_splits)), \
                      is_training=True,
                      num_epochs=args.num_train_epochs,
                      tokenizer=tokenizer)
            for step, batch in enumerate(train_dataloader):
                if global_step % args.print_every == 0:
                    logger.info("Training step: {}".format(global_step))
                global_step += 1
                batch = [b.to(torch.device("cuda")) for b in batch]
                if args.debug:
                  tok = BertTokenizer.from_pretrained(args.bert_name)
                  tok.convert_ids_to_tokens(batch[0][0][0][0].tolist())

                loss = model(batch, global_step)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if torch.isnan(loss).data:
                    logger.info("Stop training because loss=%s" % (loss.data))
                    stop_training=True
                    break
                train_losses.append(loss.detach().cpu())
                loss.backward()


                if global_step % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()    # We have accumulated enought gradients
                    if args.scheduler == "warmup":
                      scheduler.step()
                    model.zero_grad()

                if global_step % args.eval_period == 0:
                    logger.info("Evaluating")
                    model.eval()
                    evaluation, _ =  predict(logger, args, model,
                                          features, dataloader, global_step,
                                          freebase_entities=freebase_entities,
                                          write_prediction=False)
                    logger.info("Step %d Train loss %.2f %s %.2f%% (%s %.2f%%) on epoch=%d" % (
                        global_step,
                        np.mean(train_losses),
                        main_metric,
                        evaluation[main_metric]*100,
                        second_metric,
                        evaluation[second_metric]*100,
                        epoch))
                    if args.scheduler == "plateau":
                      scheduler.step(evaluation[main_metric])
                    train_losses = []
                    if best_accuracy < (evaluation[main_metric], evaluation[second_metric]):
                        model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
                        torch.save(model_state_dict, os.path.join(args.output_dir,
                                                                  "best-model-{}.pt".format(global_step)))
                        logger.info("Saving model with best: %.2f%% -> %.2f%% on epoch=%d, global_step=%d" % \
                                (best_accuracy[0]*100.0, evaluation[main_metric]*100.0, epoch, global_step))
                        best_accuracy = (evaluation[main_metric], evaluation[second_metric])
                        wait_step = 0
                        stop_training = False
                    else:

                        wait_step += 1
                        if wait_step >= args.wait_step:
                            stop_training = True
                    model.train()
            if stop_training:
                break

        logger.info("Training finished!")

    elif args.do_predict:
        if type(model)==list:
            model = [m.eval() for m in model]
        else:
            model.eval()

        setWikiTitles = None
        evaluation, _ = predict(logger, args, model, features, dataloader, freebase_entities=freebase_entities, wikiTitles=setWikiTitles)


def predict(logger, args, model, features, dataloader,
            global_step=-1, freebase_entities=None, write_prediction=True, wikiTitles=None):

    outputs = {}
    if args.verbose:
        dataloader = tqdm(dataloader)

    for batch in dataloader:
        with torch.no_grad():
            # batch contains input_ids, input_masks, indices
            # put input_ids and masks into cuda memory
            batch_to_feed = [b.cuda() for b in batch[:-1]]
            batch_switch_logits = model(batch_to_feed)
            batch_switch_logits = batch_switch_logits.detach().cpu().tolist()
            assert len(batch[0])==len(batch_switch_logits)
            for index, switch_logit in zip(batch[2], batch_switch_logits):
                outputs[index.item()] = switch_logit

    if write_prediction and args.n_paragraphs != None and len(args.n_paragraphs.split(','))>0:
        n_paragraphs = [int(n) for n in args.n_paragraphs.split(',')]
    else:
        n_paragraphs = None
    evaluation, all_results = get_qa_predictions(logger, features, outputs,
                                                    args.max_answer_length, n_paragraphs,
                                                    use_regex="trec" in args.predict_file,
                                                    freebase_entities=freebase_entities,
                                                    write_prediction=write_prediction, wikiTitles=wikiTitles)

    if write_prediction:
        output_prediction_file = os.path.join(args.output_dir,
                                              "{}predictions{}.json".format(args.prefix, '' if write_prediction else '-{}'.format(global_step)))
        with open(output_prediction_file, 'w') as f:
            for result in all_results:
                f.write(json.dumps(result)+"\n")

    return evaluation, all_results

if __name__ == "__main__":
    main()
