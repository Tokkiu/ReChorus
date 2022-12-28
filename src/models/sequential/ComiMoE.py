# -*- coding: UTF-8 -*-
# @Author  : Chenyang Wang
# @Email   : THUwangcy@gmail.com

""" ComiRec
Reference:
    "Controllable Multi-Interest Framework for Recommendation"
    Cen et al., KDD'2020.
CMD example:
    python main.py --model_name ComiRec --emb_size 64 --lr 1e-3 --l2 1e-6 --attn_size 8 --K 4 --add_pos 1 \
    --history_max 20 --dataset 'Grocery_and_Gourmet_Food'
"""

import torch
import torch.nn as nn
import numpy as np

from models.BaseModel import SequentialModel
from utils import layers


class ComiMoE(SequentialModel):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'attn_size', 'K']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--attn_size', type=int, default=8,
                            help='Size of attention vectors.')
        parser.add_argument('--K', type=int, default=2,
                            help='Number of hidden intent.')
        parser.add_argument('--add_pos', type=int, default=1,
                            help='Whether add position embedding.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.attn_size = args.attn_size
        self.K = args.K
        self.k = args.K
        self.num_experts = args.K
        self.add_pos = args.add_pos
        self.max_his = args.history_max
        self.len_range = torch.from_numpy(np.arange(self.max_his)).to(self.device)
        self.experts = nn.ModuleList([
            ComiExpert(args, corpus, k=1)
            for _ in range(self.K)
        ])

        self.primary = ComiExpert(args, corpus, k=1)
        self.noisy_gating = True
        self.w_gate = nn.Parameter(torch.zeros(self.emb_size, self.K), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(self.emb_size, self.K), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        self._define_params()
        self.apply(self.init_weights)

    def _define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        if self.add_pos:
            self.p_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)
        for expert in self.experts:
            expert.i_embeddings = self.i_embeddings
        self.primary.i_embeddings = self.i_embeddings

    def forward(self, feed_dict):
        self.check_list = []
        i_ids = feed_dict['item_id']  # [batch_size, -1]
        history = feed_dict['history_items']  # [batch_size, history_max]
        lengths = feed_dict['lengths']  # [batch_size]
        batch_size, seq_len = history.shape

        his_item_vectors = self.i_embeddings(history)

        his_vectors = [expert(history, lengths, his_item_vectors) for expert in self.experts]
        his_vectors = torch.cat(his_vectors, 1) # bsz, K, emb

        vu = self.primary(history, lengths, his_item_vectors).squeeze(1)
        gates, load = self.noisy_top_k_gating(vu, self.training)
        val, gtx = gates.topk(1)
        interest_vectors = his_vectors.gather(1, gtx.unsqueeze(2).repeat(1, 1, self.emb_size))

        i_vectors = self.i_embeddings(i_ids)
        if feed_dict['phase'] == 'train':
            target_vector = i_vectors[:, 0]  # bsz, emb
            target_pred = (interest_vectors * target_vector[:, None, :]).sum(-1)  # bsz, K
            idx_select = target_pred.max(-1)[1]  # bsz
            user_vector = interest_vectors[torch.arange(batch_size), idx_select, :]  # bsz, emb
            prediction = (user_vector[:, None, :] * i_vectors).sum(-1)
        else:
            prediction = (interest_vectors[:, None, :, :] * i_vectors[:, :, None, :]).sum(-1)  # bsz, -1, K
            prediction = prediction.max(-1)[0]  # bsz, -1

        return {'prediction': prediction.view(batch_size, -1)}


    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load



class ComiExpert(SequentialModel):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'attn_size', 'K']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--attn_size', type=int, default=8,
                            help='Size of attention vectors.')
        parser.add_argument('--K', type=int, default=2,
                            help='Number of hidden intent.')
        parser.add_argument('--add_pos', type=int, default=1,
                            help='Whether add position embedding.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus, k=0):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.attn_size = args.attn_size
        self.K = args.K
        if k > 0:
            self.K = k
        self.add_pos = args.add_pos
        self.max_his = args.history_max
        self.len_range = torch.from_numpy(np.arange(self.max_his)).to(self.device)
        self._define_params()
        self.apply(self.init_weights)

    def _define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        if self.add_pos:
            self.p_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)
        self.W1 = nn.Linear(self.emb_size, self.attn_size)
        self.W2 = nn.Linear(self.attn_size, self.K)

    def forward(self, history, lengths, his_vectors):
        self.check_list = []
        batch_size, seq_len = history.shape

        valid_his = (history > 0).long()

        if self.add_pos:
            position = (lengths[:, None] - self.len_range[None, :seq_len]) * valid_his
            pos_vectors = self.p_embeddings(position)
            his_pos_vectors = his_vectors + pos_vectors
        else:
            his_pos_vectors = his_vectors

        # Self-attention
        attn_score = self.W2(self.W1(his_pos_vectors).tanh())  # bsz, his_max, K
        attn_score = attn_score.masked_fill(valid_his.unsqueeze(-1) == 0, -np.inf)
        attn_score = attn_score.transpose(-1, -2)  # bsz, K, his_max
        attn_score = (attn_score - attn_score.max()).softmax(dim=-1)
        attn_score = attn_score.masked_fill(torch.isnan(attn_score), 0)
        interest_vectors = (his_vectors[:, None, :, :] * attn_score[:, :, :, None]).sum(-2)  # bsz, K, emb

        return interest_vectors