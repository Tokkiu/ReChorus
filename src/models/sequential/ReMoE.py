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
import torch.nn.functional as F

import torch
import torch.nn as nn
import numpy as np
from helpers.KDAReader import KDAReader
import pandas as pd

from models.BaseModel import SequentialModel
from utils import layers

from torch.distributions.normal import Normal
torch.set_printoptions(
    precision=2,    # 精度，保留小数点后几位，默认4
    threshold=np.inf,
    edgeitems=3,
    linewidth=200,  # 每行最多显示的字符数，默认80，超过则换行显示
    profile=None,
    sci_mode=False  # 用科学技术法显示数据，默认True
)
class ReMoE(SequentialModel):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'attn_size', 'K']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--attn_size', type=int, default=8,
                            help='Size of attention vectors.')
        parser.add_argument('--K', type=int, default=4,
                            help='Number of hidden intent.')
        parser.add_argument('--top', type=int, default=4,
                            help='Top k to select.')
        parser.add_argument('--add_pos', type=int, default=1,
                            help='Whether add position embedding.')
        parser.add_argument('--num_layers', type=int, default=1,
                            help='Number of self-attention layers.')
        parser.add_argument('--num_heads', type=int, default=4,
                            help='Number of attention heads.')
        parser.add_argument('--use_scaler', type=int, default=1,
                            help='scale experts by weight.')
        parser.add_argument('--moe_loss', type=float, default=0.01,
                            help='moe loss weight.')
        parser.add_argument('--pre_softmax', type=int, default=0,
                            help='pre softmax.')
        parser.add_argument('--print_batch', type=int, default=10,
                            help='pre softmax.')
        parser.add_argument('--print_seq', type=int, default=0,
                            help='pre softmax.')
        parser.add_argument('--fusion', type=str, default='fusion',
                            help='pre softmax.')

        # Reweight
        parser.add_argument('--re_atten', type=int, default=0,
                            help='reweight gates.')


        # Annealing temp for gumbel softmax
        parser.add_argument('--use_gumbel', type=int, default=0,
                            help='change_temp.')
        parser.add_argument('--temp_decay', type=float, default=0.99999,
                            help='temp_decay.')
        parser.add_argument('--max_temp', type=float, default=2.0,
                            help='temp_decay.')
        parser.add_argument('--min_temp', type=float, default=0.5,
                            help='temp_decay.')


        # Xavier init
        parser.add_argument('--xav_init', type=int, default=0,
                            help='xav_init.')

        # Atten temperature
        parser.add_argument('--atten_temp', type=float, default=1.0,
                            help='temp_decay.')
        parser.add_argument('--reg_loss_ratio', type=float, default=0.0,
                            help='reg loss.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.attn_size = args.attn_size
        self.k = args.top # top k
        self.num_experts = args.K # expert count
        self.add_pos = args.add_pos
        self.max_his = args.history_max
        self.use_scaler = args.use_scaler == 1
        self.pre_softmax = args.pre_softmax == 1
        self.loss_coef = args.moe_loss
        self.len_range = torch.from_numpy(np.arange(self.max_his)).to(self.device)
        self.max_his = args.history_max
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.fusion = args.fusion
        self.print_batch = args.print_batch
        self.print_seq = args.print_seq
        self.atten_temp = args.atten_temp

        # Temp decay
        self.max_temp, self.min_temp, self.temp_decay = (1, 1, 1)
        self.max_temp = args.max_temp # 2.0
        self.min_temp = args.min_temp # 0.5
        self.temp_decay = args.temp_decay # 0.999
        self.use_gumbel = args.use_gumbel > 0
        self.curr_temp = self.max_temp
        self.num_updates = 0

        if self.fusion not in ['fusion','top']:
            raise Exception("Invalid fusion", self.fusion)
        if self.fusion == 'fusion':
            self.k = self.num_experts


        self.noisy_gating = True
        self.w_gate = nn.Parameter(torch.zeros(self.emb_size, self.num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(self.emb_size, self.num_experts), requires_grad=True)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        self._define_params()

        # reweight gate by attention
        self.re_atten = args.re_atten > 0
        self.re_layer1 = nn.Linear(self.max_his, self.emb_size)
        self.re_layer2 = nn.Linear(self.emb_size, 1)

        # Init weights
        self.apply(self.init_weights)
        self.xav_init = args.xav_init > 0
        if self.xav_init:
            self.apply(self.xavier_normal_initialization)

        self.experts = nn.ModuleList([
            ComiExpert(args, corpus, k=1)
            for _ in range(self.num_experts)
        ])

        # share item embedding
        for expert in self.experts:
            expert.i_embeddings = self.i_embeddings
        self.primary = ComiExpert(args, corpus, k=1)
        self.primary.i_embeddings = self.i_embeddings

    def _define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.p_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)
        self.transformer_block = nn.ModuleList([
            layers.TransformerLayer(d_model=self.emb_size, d_ff=self.emb_size, n_heads=self.num_heads,
                                    dropout=self.dropout, kq_same=False)
            for _ in range(self.num_layers)
        ])

    def calculate_reg_loss(self, attention):
        C_mean = torch.mean(attention, dim=2, keepdim=True)
        C_reg = (attention - C_mean)
        # C_reg = C_reg.matmul(C_reg.transpose(1,2)) / self.hidden_size
        C_reg = torch.bmm(C_reg, C_reg.transpose(1, 2)) / self.emb_size
        if not self.training:
            print("C_reg:")
            print(C_reg[:self.print_batch].detach())
        n1 = torch.norm(C_reg, dim=(1, 2)) ** 2
        dr = torch.diagonal(C_reg, dim1=-2, dim2=-1)
        n2 = torch.norm(dr, dim=(1)) ** 2
        return (n1 - n2).sum() / 2

    def forward(self, feed_dict):
        self.check_list = []
        i_ids = feed_dict['item_id']  # [batch_size, -1]
        history = feed_dict['history_items']  # [batch_size, history_max]
        lengths = feed_dict['lengths']  # [batch_size]
        batch_size, seq_len = history.shape

        his_item_vectors = self.i_embeddings(history)

        valid_his = (history > 0).long()
        position = (lengths[:, None] - self.len_range[None, :seq_len]) * valid_his
        pos_vectors = self.p_embeddings(position)
        his_sas_vectors = his_item_vectors + pos_vectors



        # Self-attention
        causality_mask = np.tril(np.ones((1, 1, seq_len, seq_len), dtype=np.int))
        attn_mask = torch.from_numpy(causality_mask).to(self.device)
        for block in self.transformer_block:
            his_sas_vectors = block(his_sas_vectors, attn_mask)
        his_sas_vectors = his_sas_vectors * valid_his[:, :, None].float()

        # Call experts
        expert_output = [expert(history, lengths, his_sas_vectors, feed_dict, temp=self.atten_temp) for expert in self.experts]
        his_vectors = [out[0] for out in expert_output]
        his_vectors = torch.cat(his_vectors, 1)
        atten_vectors = [out[1] for out in expert_output]
        atten_vectors_logit = [out[2] for out in expert_output]
        atten_vectors = torch.cat(atten_vectors, 1)
        atten_vectors_logit = torch.cat(atten_vectors_logit, 1)

        # Call primary expert
        vu, atten, _ = self.primary(history, lengths, his_sas_vectors, feed_dict)
        vu = vu.squeeze(1)

        # Call reweighting attention
        reatten_vectors, reatten_input, reatten_vectors = None, None, None
        if self.re_atten:
            reatten_vectors = self.re_layer2(self.re_layer1(atten_vectors).tanh()).squeeze(-1)  # bsz, experts


        # Call gates
        gates, load, gate_logits = self.noisy_top_k_gating(vu, self.training, bias=reatten_vectors)
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= self.loss_coef

        if self.use_scaler:
            his_vectors = his_vectors * gates.unsqueeze(2)

        if self.fusion == 'fusion':
            interest_vectors = his_vectors.sum(1).unsqueeze(1)
        elif self.fusion == 'top':
            val, gtx = gates.topk(self.k)
            if not self.training and gtx.size(0) % 16 == 0:
                print("gtx", gtx.reshape(16, -1))
            interest_vectors = his_vectors.gather(1, gtx.unsqueeze(2).repeat(1, 1, self.emb_size))

        # Debugging inference
        if not self.training:
            if self.print_seq > 0:
                print("seqs:")
                print(history[:self.print_seq].detach())
            if self.print_batch > 0 and self.re_atten:
                print("reatten_inputs:")
                print(reatten_input[:self.print_batch].detach())
                print("reatten_logits:")
                print(reatten_vectors[:self.print_batch].detach())
            if self.print_batch > 0:
                print("lengths:")
                print(lengths[:self.print_batch].detach())
                print("attention logits:")
                print(atten_vectors_logit[:self.print_batch].detach())
                print("attention weight:")
                print(atten_vectors[:self.print_batch].detach())
                print("gate_logits:")
                print(gate_logits[:self.print_batch].detach())
                print("gates:")
                print(gates[:self.print_batch].detach())

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

        reg_loss = self.calculate_reg_loss(atten_vectors) * self.reg_loss_ratio

        return {'prediction': prediction.view(batch_size, -1), 'moe_loss':loss, 'reg_loss':reg_loss}

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def update_per_epoch(self, num_updates):
        self.curr_temp = max(
            self.max_temp * self.temp_decay ** num_updates, self.min_temp
        )
    def log_per_epoch(self):
        return "temp set " + str(self.curr_temp) if self.use_gumbel else ""

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

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2, bias=None):
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
        if self.pre_softmax:
            logits = self.softmax(logits)
        if bias is not None:
            n_logits = logits + bias
        else:
            n_logits = logits
        top_logits, top_indices = n_logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        if self.pre_softmax:
            top_k_gates = top_k_logits
        else:
            if self.use_gumbel:
                top_k_gates = F.gumbel_softmax(top_k_logits.float(), tau=self.curr_temp, hard=False).type_as(top_k_logits)
            # elif self.temp_moe:
            #     n_top_k_logits /= self.gumbel_temperature
            #     top_k_gates = self.softmax(n_top_k_logits)
            else:
                top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load, n_logits


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
        self.xav_init = args.xav_init > 0

        self.apply(self.init_weights)
        if self.xav_init:
            self.apply(self.xavier_normal_initialization)


    def _define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        if self.add_pos:
            self.p_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)
        self.W1 = nn.Linear(self.emb_size, self.attn_size)
        self.W2 = nn.Linear(self.attn_size, self.K)

    def forward(self, history, lengths, his_vectors, feed_dict, temp=1.0):
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
        attn_score = (attn_score - attn_score.max())
        # attn_score /= temp
        attn_score_out = (attn_score/temp).softmax(dim=-1).masked_fill(torch.isnan(attn_score), 0)
        interest_vectors = (his_vectors[:, None, :, :] * attn_score_out[:, :, :, None]).sum(-2)  # bsz, K, emb
        return interest_vectors, attn_score_out, attn_score
