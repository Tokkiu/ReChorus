# -*- coding: UTF-8 -*-

import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as BaseDataset
from torch.nn.utils.rnn import pad_sequence
from typing import List

from utils import utils
from helpers.BaseReader import BaseReader

from torch.nn.init import xavier_normal_, xavier_uniform_, constant_
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os, math
from collections import defaultdict


class BaseModel(nn.Module):
    reader, runner = None, None  # choose helpers in specific model classes
    extra_log_args = []

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--model_path', type=str, default='',
                            help='Model save path.')
        parser.add_argument('--buffer', type=int, default=1,
                            help='Whether to buffer feed dicts for dev/test')
        return parser

    @staticmethod
    def init_weights(m):
        if 'Linear' in str(type(m)):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif 'Embedding' in str(type(m)):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)


    def xavier_normal_initialization(self, module):
        r"""using `xavier_normal_`_ in PyTorch to initialize the parameters in
        nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
        using constant 0 to initialize.

        .. _`xavier_normal_`:
            https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_normal_#torch.nn.init.xavier_normal_

        Examples:
            >>> self.apply(xavier_normal_initialization)
        """
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def vis_emb(self, item_embedding, epoch):
        embs = item_embedding.weight.detach().cpu().numpy()
        label = []
        x_in = []
        xs = []
        for k, v in self.item_cnt.items():
            xs.append(k)
            x_in.append(embs[k])
            nv = round(math.log(v))
            label.append(nv)
        print("max label", max(label))
        epoch = '{0:03d}'.format(epoch)
        X_tsne = TSNE(n_components=2, random_state=33).fit_transform(x_in)
        plt.figure(figsize=(10, 10))
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=label, label="MoE", s=15, cmap='coolwarm')
        plt.legend()
        plt.savefig('images/'+self.name+'_tsne_' + epoch + '.png', dpi=120)


        x_inn = F.normalize(torch.tensor(np.array(x_in)),dim=1, p=2).numpy()
        X_tsne = TSNE(n_components=2, random_state=33).fit_transform(x_inn)
        plt.figure(figsize=(10, 10))
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=label, label="MoE", s=15, cmap='coolwarm')
        plt.legend()
        plt.savefig('images/' + self.name + '_tsne_n_' + epoch + '.png', dpi=120)


    def __init__(self, args, corpus: BaseReader):
        super(BaseModel, self).__init__()
        self.device = args.device
        self.model_path = args.model_path
        self.buffer = args.buffer
        self.optimizer = None
        self.check_list = list()  # observe tensors in check_list every check_epoch
        self.item_cnt = defaultdict(int)

    """
    Key Methods
    """
    def _define_params(self):
        pass

    def forward(self, feed_dict: dict) -> dict:
        """
        :param feed_dict: batch prepared in Dataset
        :return: out_dict, including prediction with shape [batch_size, n_candidates]
        """
        pass

    def loss(self, out_dict: dict) -> torch.Tensor:
        pass

    """
    Auxiliary Methods
    """
    def customize_parameters(self) -> list:
        # customize optimizer settings for different parameters
        weight_p, bias_p = [], []
        for name, p in filter(lambda x: x[1].requires_grad, self.named_parameters()):
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        optimize_dict = [{'params': weight_p}, {'params': bias_p, 'weight_decay': 0}]
        return optimize_dict

    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        utils.check_dir(model_path)
        torch.save(self.state_dict(), model_path)
        # logging.info('Save model to ' + model_path[:50] + '...')

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        self.load_state_dict(torch.load(model_path))
        logging.info('Load model from ' + model_path)

    def count_variables(self) -> int:
        total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_parameters

    def actions_after_train(self):  # e.g., save selected parameters
        pass

    """
    Define Dataset Class
    """
    class Dataset(BaseDataset):
        def __init__(self, model, corpus, phase: str):
            self.model = model  # model object reference
            self.corpus = corpus  # reader object reference
            self.phase = phase  # train / dev / test

            self.buffer_dict = dict()
            self.data = utils.df_to_dict(corpus.data_df[phase])
            # ↑ DataFrame is not compatible with multi-thread operations

        def __len__(self):
            if type(self.data) == dict:
                for key in self.data:
                    return len(self.data[key])
            return len(self.data)

        def __getitem__(self, index: int) -> dict:
            if self.model.buffer and self.phase != 'train':
                return self.buffer_dict[index]
            return self._get_feed_dict(index)

        # ! Key method to construct input data for a single instance
        def _get_feed_dict(self, index: int) -> dict:
            pass

        # Called after initialization
        def prepare(self):
            if self.model.buffer and self.phase != 'train':
                for i in tqdm(range(len(self)), leave=False, desc=('Prepare ' + self.phase)):
                    self.buffer_dict[i] = self._get_feed_dict(i)

        # Called before each training epoch (only for the training dataset)
        def actions_before_epoch(self):
            pass

        # Collate a batch according to the list of feed dicts
        def collate_batch(self, feed_dicts: List[dict]) -> dict:
            feed_dict = dict()
            for key in feed_dicts[0]:
                if isinstance(feed_dicts[0][key], np.ndarray):
                    tmp_list = [len(d[key]) for d in feed_dicts]
                    if any([tmp_list[0] != l for l in tmp_list]):
                        stack_val = np.array([d[key] for d in feed_dicts], dtype=np.object)
                    else:
                        stack_val = np.array([d[key] for d in feed_dicts])
                else:
                    stack_val = np.array([d[key] for d in feed_dicts])
                if stack_val.dtype == np.object:  # inconsistent length (e.g., history)
                    feed_dict[key] = pad_sequence([torch.from_numpy(x) for x in stack_val], batch_first=True)
                else:
                    feed_dict[key] = torch.from_numpy(stack_val)
            feed_dict['batch_size'] = len(feed_dicts)
            feed_dict['phase'] = self.phase
            return feed_dict


class GeneralModel(BaseModel):
    reader, runner = 'BaseReader', 'BaseRunner'

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--num_neg', type=int, default=1,
                            help='The number of negative items during training.')
        parser.add_argument('--dropout', type=float, default=0,
                            help='Dropout probability for each deep layer')
        parser.add_argument('--test_all', type=int, default=0,
                            help='Whether testing on all the items.')
        return BaseModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.user_num = corpus.n_users
        self.item_num = corpus.n_items
        self.num_neg = args.num_neg
        self.dropout = args.dropout
        self.test_all = args.test_all

    def log_per_epoch(self):
        return ""

    def update_per_epoch(self, n):
        return n

    def loss(self, out_dict: dict) -> torch.Tensor:
        """
        BPR ranking loss with optimization on multiple negative samples (a little different now)
        "Recurrent neural networks with top-k gains for session-based recommendations"
        :param out_dict: contain prediction with [batch_size, -1], the first column for positive, the rest for negative
        :return:
        """
        predictions = out_dict['prediction']
        pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:]
        neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
        loss = -((pos_pred[:, None] - neg_pred).sigmoid() * neg_softmax).sum(dim=1).log().mean()
        if 'moe_loss' in out_dict:
            loss += out_dict['moe_loss']
        if 'reg_loss' in out_dict:
            loss += out_dict['reg_loss']
        # neg_pred = (neg_pred * neg_softmax).sum(dim=1)
        # loss = F.softplus(-(pos_pred - neg_pred)).mean()
        # ↑ For numerical stability, use 'softplus(-x)' instead of '-log_sigmoid(x)'
        return loss

    class Dataset(BaseModel.Dataset):
        def _get_feed_dict(self, index):
            user_id, target_item = self.data['user_id'][index], self.data['item_id'][index]
            if self.phase != 'train' and self.model.test_all:
                neg_items = np.arange(1, self.corpus.n_items)
            else:
                neg_items = self.data['neg_items'][index]
            item_ids = np.concatenate([[target_item], neg_items]).astype(int)
            feed_dict = {
                'user_id': user_id,
                'item_id': item_ids
            }
            return feed_dict

        # Sample negative items for all the instances
        def actions_before_epoch(self):
            neg_items = np.random.randint(1, self.corpus.n_items, size=(len(self), self.model.num_neg))
            for i, u in enumerate(self.data['user_id']):
                clicked_set = self.corpus.train_clicked_set[u]  # neg items are possible to appear in dev/test set
                # clicked_set = self.corpus.clicked_set[u]  # neg items will not include dev/test set
                for j in range(self.model.num_neg):
                    while neg_items[i][j] in clicked_set:
                        neg_items[i][j] = np.random.randint(1, self.corpus.n_items)
            self.data['neg_items'] = neg_items


class SequentialModel(GeneralModel):
    reader = 'SeqReader'

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--history_max', type=int, default=20,
                            help='Maximum length of history.')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.history_max = args.history_max

    class Dataset(GeneralModel.Dataset):
        def __init__(self, model, corpus, phase):
            super().__init__(model, corpus, phase)
            idx_select = np.array(self.data['position']) > 0  # history length must be non-zero
            for key in self.data:
                self.data[key] = np.array(self.data[key])[idx_select]
            for item in self.data['item_id']:
                model.item_cnt[item] += 1

        def _get_feed_dict(self, index):
            feed_dict = super()._get_feed_dict(index)
            pos = self.data['position'][index]
            user_seq = self.corpus.user_his[feed_dict['user_id']][:pos]
            if self.model.history_max > 0:
                user_seq = user_seq[-self.model.history_max:]
                # user_seq += [(0, 0)] * (self.model.history_max - len(user_seq))
            feed_dict['history_items'] = np.array([x[0] for x in user_seq])
            feed_dict['history_times'] = np.array([x[1] for x in user_seq])
            feed_dict['lengths'] = len(feed_dict['history_items'])
            return feed_dict
