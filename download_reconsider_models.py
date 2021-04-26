# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import argparse

parser = argparse.ArgumentParser(description='download_reconsider_models.py')

parser.add_argument('--model', default = 'nq_bbase',
                    help='Which model to download?')
opt = parser.parse_args()


models = {
  'nq_bbase': 'http://dl.fbaipublicfiles.com/reconsider/models/ps.nq.qp_mp.nopp.title.100.M_30._nq_tbz_16_ebz_144_m_20_g8__--pad_question_--max_question_length_0_--max_pass_best-model-30500.pt',
  'nq_blarge': 'http://dl.fbaipublicfiles.com/reconsider/models/ps.nq.blarge.qp_mp.nopp.title.100.M_10.tM_5._nq_tbz_16_ebz_144_m_20_g8__--pad_question_--max_question_length_0_--max_pass_best-model-39500.pt',
  'tqa_bbase': 'http://ps.tqa.bbase.qp_mp.nopp.title.100.M_30.tM_5._nq_tbz_16_ebz_144_m_20_g8__--pad_question_--max_question_length_0_--max_pass_best-model-31000.pt',
  'tqa_blarge': 'ps.tqa.blarge.blarge.qp_mp.nopp.title.100.M_10.tM_5._nq_tbz_16_ebz_144_m_20_g8__--pad_question_--max_question_length_0_--max_pass_best-model-23500.pt',
  'trec_bbase': 'ps.trec.bbase.qp_mp.nopp.title.100.M_30.tM_5._nq_tbz_16_ebz_144_m_20_g8__--train_batch_size_4_--pad_question_--max_questio_best-model-3000.pt',
  'trec_blarge': 'ps.trec.blarge_init_blarge_on_blarge.qp_mp.nopp.title.100.M_10.tM_5._nq_tbz_16_ebz_144_m_20_g8__--pad_question_--max_question_length_0_--max_pass_best-model-1000.pt',
  'webq_bbase': 'ps.webq.bbase.qp_mp.nopp.title.100.M_30.tM_5._nq_tbz_16_ebz_144_m_20_g8__--pad_question_--max_question_length_0_--max_pass_best-model-8000.pt',
  'webq_blarge': 'dl.fbaipublicfiles.com/reconsider/models/ps.webq.blarge_init_blarge_on_blarge.qp_mp.nopp.title.100.M_10.tM_5._nq_tbz_16_ebz_144_m_20_g8__--pad_question_--max_question_length_0_--max_pass_best-model-1500.pt'
}

os.system('wget {}'.format(models[opt.model]))
