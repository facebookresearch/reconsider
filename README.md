# ReConsider

ReConsider is a re-ranking model that re-ranks the top-K (passage, answer-span) predictions of an Open-Domain QA Model like DPR (Karpukhin et al., 2020).

The technical details are described in:

```bibtex
@inproceedings{iyer2020reconsider,
 title={RECONSIDER: Re-Ranking using Span-Focused Cross-Attention for Open Domain Question Answering},
 author={Iyer, Srinivasan and Min, Sewon and Mehdad, Yashar and Yih, Wen-tau},
 booktitle={NAACL},
 year={2021}
}
```

[https://arxiv.org/abs/2010.10757](https://arxiv.org/abs/2010.10757)

## LICENSE

The majority of ReConsider is licensed under CC-BY-NC, however portions of the project are available under separate license terms: huggingface transformers and HotpotQA Utils are licensed under the Apache 2.0 license.

## Re-producing results from the paper

The ReConsider models in the paper are trained on the top-100 predictions from the DPR Retriever + Reader model (Karpukhin et al., 2020) on four datasets: NaturalQuestions, TriviaQA, Trec, and WebQ.

We outline all the steps here for NaturalQuestions, but the same steps can be followed for the other datasets.

0. Environment Setup
```
pip install -r requirements.txt
```


1. [optional] Get the top-100 retrieved passages for each question using the best DPR retriever model for the NQ train, dev, and test sets. We provide these in our repo, but alternatively, you can obtain them by training the DPR retriever from scratch (from [here](https://github.com/facebookresearch/dpr)). You can skip this entire step if you are only running ReConsider.

```
wget http://dl.fbaipublicfiles.com/reconsider/dpr_retriever_outputs/{nq|webq|trec|tqa}-{train|dev|test}-multi.json
```

2. [optional] Get the top-100 predictions from the DPR reader (Karpukhin et al., 2020) executed on the output of the DPR retriever, on the NQ train, dev, and test sets. We provide these in our repo, but alternatively, you can obtain them by training the DPR reader from scratch (from [here](https://github.com/facebookresearch/dpr)). You can skip this entire step if you are only running ReConsider.

```
wget http://dl.fbaipublicfiles.com/reconsider/dpr_reader_outputs/ttttt_{train|dev|test}.{nq|tqa|trec|webq}.{bbase|blarge}.output.nopp.title.json
```

3. [optional] Convert DPR reader predictions to the marked-passage format required by ReConsider. 
```
python prepare_marked_dataset.py --answer_json ttttt__train.{nq|tqa|trec|webq}.{bbase|blarge}.output.nopp.title.json --orig_json {nq|webq|trec|tqa}-train-multi.json --out_json paraphrase_selection_train.{nq|tqa|trec|webq}.{bbase|blarge}.100.qp_mp.nopp.title.json --train_M 100

python prepare_marked_dataset.py --answer_json ttttt_dev.{nq|tqa|trec|webq}.{bbase|blarge}.output.nopp.title.json --orig_json {nq|webq|trec|tqa}-dev-multi.json --out_json paraphrase_selection_dev.{nq|tqa|trec|webq}.{bbase|blarge}.5.qp_mp.nopp.title.json --dev --test_M 5

python prepare_marked_dataset.py --answer_json ttttt_test.{nq|tqa|trec|webq}.{bbase|blarge}.output.nopp.title.json --orig_json {nq|webq|trec|tqa}-test-multi.json --out_json paraphrase_selection_test.{nq|tqa|trec|webq}.{bbase|blarge}.5.qp_mp.nopp.title.json --dev --test_M 5
```

We also provide these files, so that you don't need to execute this command. You can directly download the output files using:
```
wget http://dl.fbaipublicfiles.com/reconsider/reconsider_inputs/paraphrase_selection_{train|dev|test}.{nq|tqa|trec|webq}.{bbase|blarge}.qp_mp.nopp.title.json
```

4. Train ReConsider Models
For Base models:
```
dset={nq|tqa|trec|webq}
python main.py --do_train --output_dir ps.$dset.bbase --train_file paraphrase_selection_train.$dset.bbase.qp_mp.nopp.title.json --predict_file paraphrase_selection_dev.$dset.bbase.qp_mp.nopp.title.json --train_batch_size 16 --predict_batch_size 144 --eval_period 500 --threads 80 --pad_question --max_question_length 0 --max_passage_length 240 --train_M 30 --test_M 5
```
For Large models:
```
dset={nq|tqa|trec|webq}
python main.py --do_train --output_dir ps.$dset.bbase --train_file paraphrase_selection_train.$dset.bbase.qp_mp.nopp.title.json --predict_file paraphrase_selection_dev.$dset.bbase.qp_mp.nopp.title.json --train_batch_size 16 --predict_batch_size 144 --eval_period 500 --threads 80 --pad_question --max_question_length 0 --max_passage_length 240 --train_M 10 --test_M 5 --bert_name bert-large-uncased
```
Note: If training on Trec or Webq, initialize the model with the model trained on NQ of the corresponding size by adding this parameter: `--checkpoint $model_nq_{bbase|blarge}`. You can either train this NQ model using the commands above, or directly download it as described below:

We also provide our pre-trained models for download, using this script:
```
python download_reconsider_models.py --model {nq|trec|tqa|webq}_{bbase|blarse}
```

5. Predict on the test set using ReConsider Models
```
python main.py --do_predict --output_dir /tmp/ --predict_file paraphrase_selection_test.{nq|trec|webq|tqa}.{bbase|blarge}.qp_mp.nopp.title.json  --checkpoint {path_to_model} --predict_batch_size 72 --threads 80 --n_paragraphs 100  --verbose --prefix test_  --pad_question --max_question_length 0 --max_passage_length 240 --predict_batch_size 72 --test_M 5 --bert_name {bert-base-uncased|bert-large-uncased}
```
