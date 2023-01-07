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
class EvoMoE(SequentialModel):
    # reader = 'SeqReader'
    reader = 'KDAReader'
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
                            help='Number of hidden intent.')
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
        parser.add_argument('--fusion', type=str, default='top',
                            help='pre softmax.')

        # Fixed temp
        parser.add_argument('--temp', type=float, default=-1.0,
                            help='gumbel_temperature.')

        # Reweight
        parser.add_argument('--re_atten', type=int, default=0,
                            help='gumbel_temperature.')


        # Annealing temp
        parser.add_argument('--temp_decay', type=float, default=0.99999,
                            help='temp_decay.')
        parser.add_argument('--max_temp', type=float, default=2.0,
                            help='temp_decay.')
        parser.add_argument('--min_temp', type=float, default=0.5,
                            help='temp_decay.')

        parser.add_argument('--change_temp', type=int, default=100000,
                            help='change_temp.')

        parser.add_argument('--use_evo', type=int, default=0,
                            help='change_temp.')

        parser.add_argument('--neg_head_p', type=float, default=0.5,
                            help='The probability of sampling negative head entity.')
        parser.add_argument('--decay_factor', type=float, default=1,
                            help='decay_factor')

        parser.add_argument('--xav_init', type=int, default=0,
                            help='xav_init.')
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.attn_size = args.attn_size
        self.k = args.top
        self.num_experts = args.K
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

        self.gumbel_temperature = 2
        self.gumbel_temperature_arg = args.temp
        self.change_temp_epoch = args.change_temp
        self.temp_moe = self.gumbel_temperature > 0

        self.max_temp, self.min_temp, self.temp_decay = (2.0, 0.5, 0.99999)
        self.max_temp = args.max_temp
        self.min_temp = args.min_temp
        self.temp_decay = args.temp_decay
        self.anneal_moe = self.max_temp > 0
        self.curr_temp = self.max_temp
        self.num_updates = 0

        self.use_evo = args.use_evo > 0
        self.decay_factor = args.decay_factor
        self.neg_head_p = args.neg_head_p


        self.xav_init = args.xav_init > 0

        self.re_atten = args.re_atten > 0

        if self.fusion not in ['fusion','top']:
            raise Exception("Invalid fusion", self.fusion)


        self.noisy_gating = True
        self.w_gate = nn.Parameter(torch.zeros(self.emb_size, self.num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(self.emb_size, self.num_experts), requires_grad=True)

        self.reweight_layers = nn.ModuleList([
            nn.Linear(self.max_his, 1)
            for _ in range(self.num_experts)
        ])

        self.re_layer1 = nn.Linear(self.max_his, self.emb_size)
        self.re_layer2 = nn.Linear(self.emb_size, 1)

        self.reweight_act = torch.sigmoid

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        self._define_params()
        self.apply(self.init_weights)

        self.experts = nn.ModuleList([
            ComiExpert(args, corpus, k=1, use_evo=self.use_evo)
            for _ in range(self.num_experts)
        ])
        if self.xav_init:
            self.apply(self.xavier_normal_initialization)
        for expert in self.experts:
            expert.i_embeddings = self.i_embeddings
        self.primary = ComiExpert(args, corpus, k=1, use_evo=self.use_evo)
        self.primary.i_embeddings = self.i_embeddings

    def _define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.p_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)
        self.transformer_block = nn.ModuleList([
            layers.TransformerLayer(d_model=self.emb_size, d_ff=self.emb_size, n_heads=self.num_heads,
                                    dropout=self.dropout, kq_same=False)
            for _ in range(self.num_layers)
        ])

    def forward(self, feed_dict):
        self.check_list = []
        if self.training and feed_dict['train_epoch'] > self.change_temp_epoch:
            self.gumbel_temperature = self.gumbel_temperature_arg
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
        # attn_mask = valid_his.view(batch_size, 1, 1, seq_len)
        for block in self.transformer_block:
            his_sas_vectors = block(his_sas_vectors, attn_mask)
        his_sas_vectors = his_sas_vectors * valid_his[:, :, None].float()

        expert_output = [expert(history, lengths, his_sas_vectors, feed_dict) for expert in self.experts]

        his_vectors = [out[0] for out in expert_output]
        his_vectors = torch.cat(his_vectors, 1)
        atten_vectors = [out[1] for out in expert_output]

        vu, atten, decay = self.primary(history, lengths, his_sas_vectors, feed_dict)
        vu = vu.squeeze(1)
        print_gates = False
        if not self.training:
            if self.print_seq > 0:
                print(history[:self.print_seq])
            # print((atten[:self.print_batch]*100).int())
            print_gates = True

        reatten_vectors = None
        if self.re_atten:
            reatten_input = torch.cat(atten_vectors, 1)
            reatten_vectors = self.re_layer2(self.re_layer1(reatten_input).tanh()).squeeze(-1)  # bsz, experts
            if not self.training:
                print("reatten_input", reatten_input[:self.print_batch])
                print("reatten_vectors", reatten_vectors[:self.print_batch])

        gates, load, gate_logits = self.noisy_top_k_gating(vu, self.training, bias=reatten_vectors)
        if not self.training:
            print("gate_logits", gate_logits[:self.print_batch])
        # import pdb; pdb.set_trace()
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= self.loss_coef


            # gates = (reatten_vectors * gates)
            # gates /= gates.sum(1).unsqueeze(1)

        # import pdb; pdb.set_trace()
        if self.use_scaler:
            his_vectors = his_vectors * gates.unsqueeze(2)

        if self.fusion == 'fusion':
            if print_gates and self.print_batch > 0:
                print(gates[:self.print_batch])
                if decay is not None:
                    print(decay.reshape(gates.size(0), -1)[:self.print_batch])
            interest_vectors = his_vectors.sum(1).unsqueeze(1)
        elif self.fusion == 'top':
            val, gtx = gates.topk(self.k)
            if not self.training and gtx.size(0) % 16 == 0:
                print(gtx.reshape(16, -1))
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

        # if self.training:
        #     self.update_per_epoch(self.num_updates)
        #     self.num_updates += 1

        return {'prediction': prediction.view(batch_size, -1), 'moe_loss':loss}

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
        return "temp set " + str(self.curr_temp)

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
            if self.temp_moe:
                logits = self.softmax(logits/self.gumbel_temperature)
            else:
                logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        if self.pre_softmax:
            top_k_gates = top_k_logits
        else:
            if bias is not None:
                top_k_logits += bias
            if self.anneal_moe:
                top_k_gates = F.gumbel_softmax(top_k_logits.float(), tau=self.curr_temp, hard=False).type_as(top_k_logits)
            elif self.temp_moe:
                top_k_logits /= self.gumbel_temperature
                top_k_gates = self.softmax(top_k_logits)
            else:
                top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load, top_k_logits


    class Dataset(SequentialModel.Dataset):
        def __init__(self, model, corpus, phase):
            super().__init__(model, corpus, phase)
            if self.phase == 'train':
                self.kg_data, self.neg_heads, self.neg_tails = None, None, None

            # Prepare item-to-value dict
            item_val = self.corpus.item_meta_df.copy()
            item_val[self.corpus.item_relations] = 0  # set the value of natural item relations to None
            for idx, r in enumerate(self.corpus.attr_relations):
                base = self.corpus.n_items + np.sum(self.corpus.attr_max[:idx])
                item_val[r] = item_val[r].apply(lambda x: x + base).astype(int)
            item_vals = item_val[self.corpus.relations].values  # this ensures the order is consistent to relations
            self.item_val_dict = dict()
            for item, vals in zip(item_val['item_id'].values, item_vals.tolist()):
                self.item_val_dict[item] = [0] + vals  # the first dimension None for the virtual relation

        def _get_feed_dict(self, index):
            feed_dict = super()._get_feed_dict(index)
            feed_dict['item_val'] = [self.item_val_dict[item] for item in feed_dict['item_id']]
            delta_t = self.data['time'][index] - feed_dict['history_times']
            feed_dict['history_delta_t'] = KDAReader.norm_time(delta_t, self.corpus.t_scalar)
            if self.phase == 'train':
                feed_dict['head_id'] = np.concatenate([[self.kg_data['head'][index]], self.neg_heads[index]])
                feed_dict['tail_id'] = np.concatenate([[self.kg_data['tail'][index]], self.neg_tails[index]])
                feed_dict['relation_id'] = self.kg_data['relation'][index]
                feed_dict['value_id'] = self.kg_data['value'][index]
            return feed_dict

        def generate_kg_data(self) -> pd.DataFrame:
            rec_data_size = len(self)
            replace = (rec_data_size > len(self.corpus.relation_df))
            kg_data = self.corpus.relation_df.sample(n=rec_data_size, replace=replace).reset_index(drop=True)
            kg_data['value'] = np.zeros(len(kg_data), dtype=int)  # default for None
            tail_select = kg_data['tail'].apply(lambda x: x < self.corpus.n_items)
            item_item_df = kg_data[tail_select]
            item_attr_df = kg_data.drop(item_item_df.index)
            item_attr_df['value'] = item_attr_df['tail'].values

            sample_tails = list()  # sample items sharing the same attribute
            for head, val in zip(item_attr_df['head'].values, item_attr_df['tail'].values):
                share_attr_items = self.corpus.share_attr_dict[val]
                tail_idx = np.random.randint(len(share_attr_items))
                sample_tails.append(share_attr_items[tail_idx])
            item_attr_df['tail'] = sample_tails
            kg_data = pd.concat([item_item_df, item_attr_df], ignore_index=True)
            return kg_data

        def actions_before_epoch(self):
            super().actions_before_epoch()
            self.kg_data = self.generate_kg_data()
            heads, tails = self.kg_data['head'].values, self.kg_data['tail'].values
            relations, vals = self.kg_data['relation'].values, self.kg_data['value'].values
            self.neg_heads = np.random.randint(1, self.corpus.n_items, size=(len(self.kg_data), self.model.num_neg))
            self.neg_tails = np.random.randint(1, self.corpus.n_items, size=(len(self.kg_data), self.model.num_neg))
            for i in range(len(self.kg_data)):
                item_item_relation = (tails[i] <= self.corpus.n_items)
                for j in range(self.model.num_neg):
                    if np.random.rand() < self.model.neg_head_p:  # sample negative head
                        tail = tails[i] if item_item_relation else vals[i]
                        while (self.neg_heads[i][j], relations[i], tail) in self.corpus.triplet_set:
                            self.neg_heads[i][j] = np.random.randint(1, self.corpus.n_items)
                        self.neg_tails[i][j] = tails[i]
                    else:  # sample negative tail
                        head = heads[i] if item_item_relation else self.neg_tails[i][j]
                        tail = self.neg_tails[i][j] if item_item_relation else vals[i]
                        while (head, relations[i], tail) in self.corpus.triplet_set:
                            self.neg_tails[i][j] = np.random.randint(1, self.corpus.n_items)
                            head = heads[i] if item_item_relation else self.neg_tails[i][j]
                            tail = self.neg_tails[i][j] if item_item_relation else vals[i]
                        self.neg_heads[i][j] = heads[i]


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

    def __init__(self, args, corpus, k=0, use_evo = False):
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
        self.use_evo = use_evo
        self.decay_factor = args.decay_factor
        self.apply(self.init_weights)
        if self.use_evo:
            self.relation_num = corpus.n_relations
            # self.relation_num = 1
            self.freq_x = corpus.freq_x
            self.freq_dim = args.n_dft // 2 + 1
            self.freq_rand = args.freq_rand
            self.freq_real = nn.Embedding(self.relation_num, self.freq_dim)
            self.freq_imag = nn.Embedding(self.relation_num, self.freq_dim)
            freq = np.linspace(0, 1, self.freq_dim) / 2.
            self.freqs = torch.from_numpy(np.concatenate((freq, -freq))).to(self.device).float()
            self.relation_range = torch.from_numpy(np.arange(self.relation_num)).to(self.device)
            self.apply(self.init_weights)
            dft_freq_real = torch.tensor(np.real(self.freq_x))  # R * n_freq
            dft_freq_imag = torch.tensor(np.imag(self.freq_x))
            self.freq_real.weight.data.copy_(dft_freq_real)
            self.freq_imag.weight.data.copy_(dft_freq_imag)


    def _define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        if self.add_pos:
            self.p_embeddings = nn.Embedding(self.max_his + 1, self.emb_size)
        self.W1 = nn.Linear(self.emb_size, self.attn_size)
        self.W2 = nn.Linear(self.attn_size, self.K)

    def forward(self, history, lengths, his_vectors, feed_dict):
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
        # attn_score_out = attn_score.softmax(dim=-1).masked_fill(torch.isnan(attn_score), 0)

        decay = None
        if self.use_evo:
            batch_size, seq_len = history.shape
            valid_mask = (history > 0).view(batch_size, 1, seq_len, 1)
            # shift masked softmax
            attention = attn_score.reshape(batch_size, 1, seq_len, 1)
            attention = attention - attention.max()
            attention = attention.masked_fill(valid_mask == 0, -np.inf).softmax(dim=-2)
            # temporal evolution
            delta_t_n = feed_dict['history_delta_t'].float()  # B * H
            decay = self.idft_decay(delta_t_n).clamp(0, 1).unsqueeze(1).masked_fill(valid_mask == 0, 0.) # B * 1 * H * R
            decay = decay.mean(-1).unsqueeze(-1)
            # import pdb; pdb.set_trace()
            # attn_score = (attention * decay).squeeze(-1)
            attn_score = attn_score + decay.squeeze(-1) * self.decay_factor

        attn_score_out = attn_score.softmax(dim=-1).masked_fill(torch.isnan(attn_score), 0)

        interest_vectors = (his_vectors[:, None, :, :] * attn_score_out[:, :, :, None]).sum(-2)  # bsz, K, emb

        return interest_vectors, attn_score_out, decay
    def idft_decay(self, delta_t):
        real, imag = self.freq_real(self.relation_range), self.freq_imag(self.relation_range)
        # create conjugate symmetric to ensure real number output
        x_real = torch.cat([real, real], dim=-1)
        x_imag = torch.cat([imag, -imag], dim=-1)
        w = 2. * np.pi * self.freqs * delta_t.unsqueeze(-1)  # B * H * n_freq
        real_part = w.cos()[:, :, None, :] * x_real[None, None, :, :]  # B * H * R * n_freq
        imag_part = w.sin()[:, :, None, :] * x_imag[None, None, :, :]
        decay = (real_part - imag_part).mean(dim=-1) / 2.  # B * H * R
        return decay.float()