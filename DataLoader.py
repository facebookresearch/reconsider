# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import time
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

class MyQADataset(Dataset):
    def __init__(self,
                 positive_input_ids, positive_input_mask,
                 positive_start_positions=None, positive_end_positions=None, positive_answer_mask=None,
                 negative_input_ids=None, negative_input_mask=None,
                 is_training=False, train_M=None, test_M=None):
        self.positive_input_ids = positive_input_ids
        self.positive_input_mask = positive_input_mask
        self.positive_start_positions = positive_start_positions
        self.positive_end_positions = positive_end_positions
        self.positive_answer_mask = positive_answer_mask
        self.negative_input_ids = negative_input_ids
        self.negative_input_mask = negative_input_mask
        assert len(positive_input_ids)==len(positive_input_mask)
        assert (not is_training) or len(negative_input_ids)==len(negative_input_mask)
        self.is_qa = self.positive_start_positions is not None
        if self.is_qa:
            assert len(positive_input_ids)==len(positive_input_mask)==\
                len(positive_start_positions)==len(positive_end_positions)==len(positive_answer_mask)
        self.is_training = is_training
        self.train_M = train_M
        self.test_M = test_M
        if not is_training:
            self.indices = torch.arange(len(self.positive_input_ids), dtype=torch.long)

    def __len__(self):
        return len(self.positive_input_ids)

    # This method is called by dataloader #batch_size times, and its output is concatenated, and becomes a batch
    def __getitem__(self, idx):
        if not self.is_training:
            input_ids = self.positive_input_ids[idx][:self.test_M]
            input_mask = self.positive_input_mask[idx][:self.test_M]

            while len(input_ids) < self.test_M: #Hack - if retrieval file doesnt have enough entries to equal test_M then just concat with itself till it equals test_M 
                input_ids = torch.cat([input_ids, input_ids], dim=0)[:self.test_M]
                input_mask = torch.cat([input_mask, input_mask], dim=0)[:self.test_M]

            return input_ids, input_mask, self.indices[idx]

        #idx is the id of a training question

        # sample positive
        positive_idx = np.random.choice(len(self.positive_input_ids[idx]))
        positive_input_ids = self.positive_input_ids[idx][positive_idx]  # a randomly chosen positive answer for this question
        positive_input_mask = self.positive_input_mask[idx][positive_idx]
        if self.is_qa:
            positive_start_positions = self.positive_start_positions[idx][positive_idx]
            positive_end_positions = self.positive_end_positions[idx][positive_idx]
            positive_answer_mask = self.positive_answer_mask[idx][positive_idx]

        # sample negatives
        negative_input_ids = []
        negative_input_mask = []
        if self.train_M > 0: # Allow for train_M = 0
          negative_idxs = np.random.permutation(range(len(self.negative_input_ids[idx])))[:self.train_M-1]    
          negative_input_ids = [self.negative_input_ids[idx][i] for i in negative_idxs] # M - 1 negaitve passages
          negative_input_mask = [self.negative_input_mask[idx][i] for i in negative_idxs]

          while len(negative_input_ids)<self.train_M-1:
              negative_input_ids.append(torch.zeros_like(positive_input_ids))
              negative_input_mask.append(torch.zeros_like(positive_input_mask))

        # aggregate
        # A torch tensor with 1 positive and M - 1 negatives for 1 training question
        input_ids = torch.cat([p.unsqueeze(0) for p in [positive_input_ids] + negative_input_ids],
                                dim=0)
        input_mask = torch.cat([p.unsqueeze(0) for p in [positive_input_mask] + negative_input_mask],
                                dim=0)

        if self.is_qa:
            M = positive_start_positions.size(0) # Usually 10, coz we have 10 max answers by default
            start_positions = torch.cat([positive_start_positions.unsqueeze(0),
                                        torch.zeros((max(self.train_M-1, 0), M), dtype=torch.long)],
                                        dim=0)
            end_positions = torch.cat([positive_end_positions.unsqueeze(0),
                                    torch.zeros(( max(self.train_M-1, 0), M), dtype=torch.long)],
                                    dim=0)
            # Doing max here to avoid a bug when train_M = 0
            answer_mask = torch.cat([positive_answer_mask.unsqueeze(0),
                                    torch.zeros(( max(self.train_M-1, 0), M), dtype=torch.long)],
                                    dim=0)
            return input_ids, input_mask, start_positions, end_positions, answer_mask

        return input_ids, input_mask



class MyQADataLoader(DataLoader):

    def __init__(self, args, features, batch_size, is_training):
        if is_training:
            positive_input_ids = [torch.tensor(f['positive_input_ids'], dtype=torch.long) \
                                    for f in features]
            positive_input_mask = [torch.tensor(f['positive_input_mask'], dtype=torch.long) \
                                    for f in features]
            negative_input_ids = [torch.tensor(f['negative_input_ids'], dtype=torch.long) \
                                    for f in features]
            negative_input_mask = [torch.tensor(f['negative_input_mask'], dtype=torch.long) \
                                    for f in features]
            positive_start_positions = [torch.tensor(f['positive_start_positions'], dtype=torch.long) \
                                        for f in features]
            positive_end_positions = [torch.tensor(f['positive_end_positions'], dtype=torch.long) \
                                        for f in features]
            positive_answer_mask = [torch.tensor(f['positive_answer_mask'], dtype=torch.long) \
                                        for f in features]
        else:
            positive_input_ids = [torch.tensor(f['positive_input_ids'][:args.test_M], dtype=torch.long) \
                                    for f in features]
            positive_input_mask = [torch.tensor(f['positive_input_mask'][:args.test_M], dtype=torch.long) \
                                    for f in features]
            negative_input_ids, negative_input_mask = None, None
            positive_start_positions, positive_end_positions, positive_answer_mask = None, None, None

        dataset = MyQADataset(
            positive_input_ids=positive_input_ids,
            positive_input_mask=positive_input_mask,
            positive_start_positions=positive_start_positions,
            positive_end_positions=positive_end_positions,
            positive_answer_mask=positive_answer_mask,
            negative_input_ids=negative_input_ids,
            negative_input_mask=negative_input_mask,
            is_training=is_training,
            train_M=args.train_M,
            test_M=args.test_M)

        if is_training:
            sampler=RandomSampler(dataset)
        else:
            sampler=SequentialSampler(dataset)

        super(MyQADataLoader, self).__init__(dataset, sampler=sampler, batch_size=batch_size)


