# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from pytorch_transformers import (WEIGHTS_NAME,
                                  PreTrainedModel,
                                  BertConfig,
                                  RobertaConfig)
from pytorch_transformers.modeling_bert import (BertPreTrainedModel,
                                                BertModel)
from pytorch_transformers.modeling_roberta import RobertaModel

class QAModel(BertPreTrainedModel):
    def __init__(self, config, device, bert_name, loss_type='mml', tau=None):
        super(QAModel, self).__init__(config)
        if bert_name.startswith('bert'):
            self.bert = BertModel(config)
        elif bert_name.startswith('roberta'):
            self.bert = RobertaModel(config)
        else:
            raise NotImplementedError()
        self.qa_classifier = nn.Linear(config.hidden_size, 1)
        self.device = device
        self.apply(self.init_weights)
        self.loss_type = loss_type

    def forward(self, batch, global_step=-1):
        # batch[0] is the inputs
        # batch[1] is the attention mask
        # batch[2] is the start positions
        # batch[3] is the end positions
        # batch[4] is the answer mask

        N, M, L = batch[0].size() # N is the batch size, M is 1 +ive example and M-1 -ive examples, L is the sequence length (question + passage)?

        switch_logits = self._forward(batch[0].view(N*M, L), batch[1].view(N*M, L))

        if global_step>-1:
            return self.get_loss(switch_logits, N, M)
        else:
            # This is done during prediction
            return switch_logits.view(N, M)

    def _forward(self, input_ids, attention_mask):
        sequence_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        switch_logits = self.qa_classifier(sequence_output[:,0,:])
        return switch_logits

    def get_loss(self, switch_logits, N, M):
        #ignored_index = start_logits.size(1) # start_logits is #N x M * L, so ignored index is M?
        loss_fct = CrossEntropyLoss(reduce=False)

        # compute switch loss
        switch_logits = switch_logits.view(N, M)
        switch_labels = torch.zeros(N, dtype=torch.long).cuda() # These are indexes of the right passages. The 0th passage is always the right passage
        switch_loss = torch.sum(loss_fct(switch_logits, switch_labels)) # Get the mean of  loss of the 0th passage for each question

        return switch_loss
