#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import *
import torch.nn.init as init

import numpy as np

import sys
sys.path.append('tools')
sys.path.append('models')
import model_function
import parse, py_op






class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, d_model, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(d_model, 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        # self.softmax = BottleSoftmax()
        self.softmax = nn.Softmax(2)
        self.pooling = nn.AdaptiveMaxPool1d(1)

    def forward(self, q, k, v, attn_mask=None):

        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper

        if attn_mask is not None:

            assert attn_mask.size() == attn.size(), \
                    'Attention mask shape {} mismatch ' \
                    'with Attention logit tensor shape ' \
                    '{}.'.format(attn_mask.size(), attn.size())

            attn.data.masked_fill_(attn_mask, -float('inf'))

        attn = self.softmax(attn)

        pattn = self.pooling(attn)
        assert pattn.size()[2] == 1
        pattn = pattn.expand(attn.size())
        mask = np.zeros(attn.size(), dtype=np.float32)
        mask[pattn.data.cpu().numpy() == attn.data.cpu().numpy()] = 1
        mask = Variable(torch.from_numpy(mask).cuda())
        attn = attn * mask

        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_ks = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_vs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_v))

        self.attention = ScaledDotProductAttention(d_model)
        self.proj = nn.Linear(n_head*d_v, d_model)

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model * 2, d_model )

        init.xavier_normal(self.w_qs)
        init.xavier_normal(self.w_ks)
        init.xavier_normal(self.w_vs)

    def forward(self, q, k, v, attn_mask=None):

        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        residual = q

        # print 'q.size', q.size()
        mb_size, len_q, d_model = q.size()
        mb_size, len_k, d_model = k.size()
        mb_size, len_v, d_model = v.size()

        # treat as a (n_head) size batch
        q_s = q.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_q) x d_model
        k_s = k.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_k) x d_model
        v_s = v.repeat(n_head, 1, 1).view(n_head, -1, d_model) # n_head x (mb_size*len_v) x d_model

        # treat the result as a (n_head * mb_size) size batch
        q_s = torch.bmm(q_s, self.w_qs).view(-1, len_q, d_k)   # (n_head*mb_size) x len_q x d_k
        k_s = torch.bmm(k_s, self.w_ks).view(-1, len_k, d_k)   # (n_head*mb_size) x len_k x d_k
        v_s = torch.bmm(v_s, self.w_vs).view(-1, len_v, d_v)   # (n_head*mb_size) x len_v x d_v

        # perform attention, result size = (n_head * mb_size) x len_q x d_v
        # outputs, attns = self.attention(q_s, k_s, v_s, attn_mask=attn_mask.repeat(n_head, 1, 1))
        outputs, attns = self.attention(q_s, k_s, v_s)

        # back to original mb_size batch, result size = mb_size x len_q x (n_head*d_v)
        outputs = torch.cat(torch.split(outputs, mb_size, dim=0), dim=-1) 

        # project back to residual size
        outputs = self.proj(outputs)
        outputs = self.dropout(outputs)


        outputs = self.linear(torch.cat((residual, outputs), 2))

        return outputs, attns.data.cpu().numpy()
        # return outputs + residual


def value_embedding_data(d = 512, split = 100):
    vec = np.array([np.arange(split) * i for i in range(int(d/2))], dtype=np.float32).transpose()
    vec = vec / vec.max() 
    embedding = np.concatenate((np.sin(vec), np.cos(vec)), 1)
    embedding[0, :d] = 0
    embedding = torch.from_numpy(embedding)
    return embedding


class Attention(nn.Module):
    def __init__(self, args):
        super ( Attention, self ).__init__ ( )
        self.args = args
        self.vital_embedding = nn.Embedding (args.vocab_size, args.embed_size ) 

        self.value_embedding = nn.Embedding.from_pretrained(value_embedding_data(args.embed_size, args.n_split + 1))
        self.xv_mapping = nn.Sequential( nn.Linear (2 * args.embed_size, args.embed_size ) , nn.ReLU(), nn.Linear (args.embed_size, args.embed_size ) )

        # self.q_embedding = nn.Embedding (args.vocab_size, args.embed_size ) 
        # self.k_embedding = nn.Embedding (args.vocab_size, args.embed_size ) 
        # self.v_embedding = nn.Embedding (args.vocab_size, args.embed_size ) 
        self.q_mapping = nn.Sequential( nn.Linear (args.embed_size, args.embed_size ) , nn.ReLU(), nn.Linear (args.embed_size, args.embed_size ) )
        self.k_mapping = nn.Sequential( nn.Linear (args.embed_size, args.embed_size ) , nn.ReLU(), nn.Linear (args.embed_size, args.embed_size ) )
        self.v_mapping = nn.Sequential( nn.Linear (args.embed_size, args.embed_size ) , nn.ReLU(), nn.Linear (args.embed_size, args.embed_size ) )
        self.attention_mapping = nn.Sequential( nn.Linear (args.embed_size, args.embed_size ) , nn.ReLU(), nn.Linear (args.embed_size, args.embed_size ) )
        self.paw = nn.Sequential( nn.Linear (args.embed_size, args.embed_size ) , nn.ReLU(), nn.Linear (args.embed_size, 1), nn.Softmax(1))
        self.sigmoid = nn.Sigmoid()
        self.master_embedding= nn.Linear(args.master_size, args.embed_size ) 
        self.time_encoding = nn.Sequential (
            nn.Embedding.from_pretrained(model_function.time_encoding_data(args.embed_size, args.time_range)),
            nn.Linear ( args.embed_size, args.embed_size),
            nn.ReLU ( ),
            nn.Dropout ( 0.25 ),
            nn.Linear ( args.embed_size, args.embed_size)
            )
        self.relu = nn.ReLU ( )

        if args.use_glp:
            self.linears = nn.Sequential (
                nn.Linear ( args.hidden_size * 2, args.rnn_size ),
                nn.ReLU ( ),
                nn.Dropout ( 0.25 ),
                nn.Linear ( args.rnn_size, 1),
            )
        else:
            self.linears = nn.Linear(args.hidden_size * 2, 1, bias=False)
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.pooling_with_indices = nn.AdaptiveMaxPool1d(1, return_indices = True)
        self.att_list = [MultiHeadAttention(1, args.embed_size, 128, 128)  for _ in range(args.num_layers)]
        if args.gpu:
            self.att_list = [x.cuda() for x in self.att_list]
        self.linear = nn.Linear ( args.embed_size, 1, bias=False)
        self.linear_time = nn.Sequential(
				nn.Linear ( args.embed_size, args.embed_size),
                nn.ReLU ( ),
				nn.Linear ( args.embed_size, args.embed_size),
                nn.Dropout ( 0.1 ),
                nn.ReLU ( )
				)

    def visit_pooling(self, output):
        size = output.size()
        output = output.view(size[0] * size[1], size[2], output.size(3)) # (bs*98, 72, 512)
        output = torch.transpose(output, 1,2).contiguous() # (bs*98, 512, 72)
        output = self.pooling(output)
        output = output.view(size[0], size[1], size[3])
        return output

    def max_pooling_with_dim(self, x, dim):
        size = list(x.size())
        assert dim >= 0
        assert dim < len(size)
        s1, s3 = 1, 1
        for d in range(dim):
            s1 *= size[d]
        for d in range(dim + 1, len(size)):
            s3 *= size[d]
        x = x.view(s1, size[dim], s3)
        x = torch.transpose(x, 1,2).contiguous()
        x, idx = self.pooling_with_indices(x)
        # x = self.pooling(x)
        new_size = size[:dim] + size[dim+1:]
        x = x.view(new_size)
        return x, idx


    def attend_demo(self, xv, m, x):
        if self.args.use_value_embedding:
            x = xv[:, :, 0, :].contiguous()
            v = xv[:, :, 1, :].contiguous()
            x_size = list(x.size())                         # (bs, n_visit, n_code)
            x = x.view((x_size[0], -1))                     # (bs, n_visit * n_code)
            v = v.view((x_size[0], -1))                     # (bs, n_visit * n_code)
            x = self.vital_embedding(x)                     # (bs, n_visit * n_code, 512)
            v = self.value_embedding(v)
            x = self.xv_mapping(torch.cat((x,v), 2)).contiguous()
        else:
            x_size = list(x.size())                         # (bs, n_visit, n_code)
            x = x.view((x_size[0], -1))                     # (bs, n_visit * n_code)
            x = self.vital_embedding(x)

        m = self.master_embedding(m)                    # (bs, 512)
        m = m.view((m.size(0), 1, m.size(1)))                  # (bs, 1, 512)
        m = m.expand(x.size())                            # (bs, n_visit * n_code, 512)
        k = self.attention_mapping(x)                   # (bs, n_visit * n_code, 512)
        a = self.sigmoid(k * m)                         # (bs, n_visit * n_code, 512)
        x = x + a * m                                   # (bs, n_visit * n_code, 512)
        # x = x.view(x_size + [-1])                       # (bs, n_visit * n_code, 512)
        return x

    def attend_pattern(self, x):
        paw = self.paw(x)                               # (bs, n_visit * n_code, 1)
        # print('paw', paw.data.cpu().numpy().sum())
        # paw = paw.expand(x.size())
        # x = paw * x
        paw = paw.transpose(1,2)                        # (bs, 1, n_visit * n_code)
        # print('paw.size', paw.size())
        # print('x.size', x.size())
        x = torch.bmm(paw, x)                           # (bs, 1, 512)
        # print('x.size', x.size())
        x = x.view((x.size(0), -1))
        # print('x.size', x.size())
        return x, paw




    def forward(self, x, master, mask=None, time=None, phase='train', value=None, trend=None):
        args = self.args
        x_size = list(x.size())                        # (bs, n_visit, n_code)

        a_x = self.attend_demo(value, master, x)               # (bs, n_visit * n_code, 512 )
        q_x = self.q_mapping(a_x)                       # (bs, n_visit * n_code, 512 )
        k_x = self.k_mapping(a_x)                       # (bs, n_visit * n_code, 512 )
        v_x = self.v_mapping(a_x)                       # (bs, n_visit * n_code, 512 )


        # add time info
        time = - time.long()
        e_t = self.time_encoding(time)                # (bs, n_visit, 512)


        e_t = e_t.unsqueeze(2).contiguous()
        e_t = e_t.expand(x_size + [e_t.size(3)]).contiguous()
        # print('e_t', e_t.size())
        e_t = e_t.view(x_size[0], -1, args.embed_size)    # (bs, n_visit * n_code, 512 )
        # print('e_t', e_t.size())
        q_x = q_x.view(x_size[0], -1, args.embed_size)    # (bs, n_visit * n_code, 512 )
        k_x = k_x.view(x_size[0], -1, args.embed_size)    # (bs, n_visit * n_code, 512 )
        v_x = v_x.view(x_size[0], -1, args.embed_size)    # (bs, n_visit * n_code, 512 )


        q_x = q_x + e_t
        attn_list = []
        for i_a, att in enumerate(self.att_list):
            k_x = k_x + e_t
            k_x, attn = att(q_x, k_x, v_x)
            attn_list.append(attn)

        if args.use_gp:
            mout, idx = self.max_pooling_with_dim(k_x, 1)
            out = self.linear(mout)
            return [out]
        else:
            out, paw = self.attend_pattern(k_x)
            out = self.linear(out)
            return [out]
