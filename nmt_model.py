# -*- coding: utf-8 -*-

from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from model_embeddings import ModelEmbeddings
Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

class NMT(nn.Module):
    """ 简单的神经机器翻译模型:
        - 双向 LSTM Encoder
        - 单向 LSTM Decoder
        - 全局注意力模型
    """

    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2):
        """ 初始化 NMT 模型.
        @param embed_size (int): Embedding size (dimensionality)
        @param hidden_size (int): Hidden Size (dimensionality)
        @param vocab (Vocab): 词总述，包括 src and tgt
        @param dropout_rate (float): Dropout probability, for attention
        """
        super(NMT, self).__init__()
        self.model_embeddings = ModelEmbeddings(embed_size, vocab)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab

        # 默认值
        self.encoder = None
        self.decoder = None
        self.h_projection = None
        self.c_projection = None
        self.att_projection = None
        self.combined_output_projection = None
        self.target_vocab_projection = None
        self.dropout = None

        # 初始化各层次
        self.encoder = nn.LSTM(embed_size,self.hidden_size,dropout=self.dropout_rate,bidirectional=True) # LSTM层 输入词嵌入，输出隐藏状态
        self.decoder = nn.LSTMCell(embed_size + self.hidden_size,self.hidden_size) # LSTMCell 输入词嵌入与隐藏状态连接，输出隐藏状态
        self.h_projection = nn.Linear(self.hidden_size * 2,self.hidden_size,bias=False) # 降维2h->h
        self.c_projection = nn.Linear(self.hidden_size * 2,self.hidden_size,bias=False) # 降维2h->h
        self.att_projection = nn.Linear(self.hidden_size * 2,self.hidden_size,bias=False) # 降维2h->h
        self.combined_output_projection = nn.Linear(self.hidden_size * 3,self.hidden_size,bias=False) # 降维3h->h
        self.target_vocab_projection = nn.Linear(self.hidden_size,len(self.vocab.tgt),bias=False) # 输出投影到词库
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, source: List[List[str]], target: List[List[str]]) -> torch.Tensor:
        """ 取一个mini-batch的源句子和目标句子, 在NMT系统下学习的语言模型，计算目标句子的似然对数

        @param source (List[List[str]]): 源句子列表
        @param target (List[List[str]]): 目标句子列表, 被 `<s>` 和 `</s>` 包裹

        @returns scores (Tensor): 形状 (b, ) 的变量或张量，表示对输入的batch的每个例子，标准目标句子的似然对数
                                  这里 b = batch size.
        """

        # 计算每个源句子的长度
        source_lengths = [len(s) for s in source]

        # 把列表转换成向量
        source_padded = self.vocab.src.to_input_tensor(source, device=self.device)   # Tensor: (src_len, b)
        target_padded = self.vocab.tgt.to_input_tensor(target, device=self.device)   # Tensor: (tgt_len, b)

        # 运行网络的前向传播
        enc_hiddens, dec_init_state = self.encode(source_padded, source_lengths)
        enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)
        combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded)
        P = F.log_softmax(self.target_vocab_projection(combined_outputs), dim=-1)

        # 目标文本概率归零化
        target_masks = (target_padded != self.vocab.tgt['<pad>']).float()

        # 计算生成真实的目标文本的概率对数
        target_gold_words_log_prob = torch.gather(P, index=target_padded[1:].unsqueeze(-1), dim=-1).squeeze(-1) * target_masks[1:]
        scores = target_gold_words_log_prob.sum(dim=0)
        return scores


    def encode(self, source_padded: torch.Tensor, source_lengths: List[int]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """ 在源句子上应用encoder来得到encoder隐藏状态
            然后，取出encoder最后的状态，把他们投影成decoder初始状态
        @param source_padded (Tensor): 形状 (src_len, b) 的填充好的源句子的张量,
                                        b = batch_size, src_len = 源句子的最大长度.
                                        已按照最长到最短长度排序
        @param source_lengths (List[int]): batch 中每个源句子的实际长度列表

        @returns enc_hiddens (Tensor): 形状 (b, src_len, h*2) 的隐藏单元张量,
                                        b = batch size, src_len = 源句子的最大长度, h = hidden size.
        @returns dec_init_state (tuple(Tensor, Tensor)): 表示 decoder 的初始隐藏状态和细胞状态的张量元组
        """
        enc_hiddens, dec_init_state = None, None

        X = self.model_embeddings.source(source_padded) # 输入经过embedding层
        X = pack_padded_sequence(X,source_lengths) # 应用pack_padded_sequence
        enc_hiddens,(last_hidden,last_cell) = self.encoder(X) # encoder
        enc_hiddens = pad_packed_sequence(enc_hiddens,batch_first=True)[0] # (b, src_len, h*2)
        # 连接Encoder得到的前向和反向的隐藏状态or细胞状态,然后用线性层初始化Decoder隐藏状态or细胞状态
        init_decoder_hidden = self.h_projection(torch.cat((last_hidden[0],last_hidden[1]),dim=1))
        init_decoder_cell = self.c_projection(torch.cat((last_cell[0],last_cell[1]),dim=1))
        dec_init_state = (init_decoder_hidden,init_decoder_cell) # 形成元组

        return enc_hiddens, dec_init_state


    def decode(self, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor,
                dec_init_state: Tuple[torch.Tensor, torch.Tensor], target_padded: torch.Tensor) -> torch.Tensor:
        """对每个 batch 计算连接的输出向量
        @param enc_hiddens (Tensor): 隐藏状态 (b, src_len, h*2),
                                     b = batch size, src_len = 源句子的最大长度, h = hidden size.
        @param enc_masks (Tensor): 句子掩码张量 (b, src_len),
                                     b = batch size, src_len = 源句子的最大长度.
        @param dec_init_state (tuple(Tensor, Tensor)): deocder 初始的隐藏状态和细胞状态
        @param target_padded (Tensor): 标准填充好的目标句子 (tgt_len, b),
                                       tgt_len = 目标句子的最大长度, b = batch size.

        @returns combined_outputs (Tensor): 连接输出的张量  (tgt_len, b,  h),
                                        tgt_len = 目标句子的最大长度, b = batch_size,  h = hidden size
        """
        # 在最大长度的句子中去掉<END>标识
        target_padded = target_padded[:-1]

        # 初始化解码器状态(隐藏和细胞)
        dec_state = dec_init_state

        # 初始化上一步连接好的输出向量 o_{t-1}为0
        batch_size = enc_hiddens.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

        # 初始化一个列表，用于收集每一个时间步连接好的输出向量 o_t
        combined_outputs = []

        enc_hiddens_proj = self.att_projection(enc_hiddens)  # (b, src_len, h)   计算 W_attProj·h_enc
        Y = self.model_embeddings.target(target_padded)  # (tgt_len, b, e)
        for Y_t in torch.split(Y, 1):  # (1, b, e) 在0维度上拆分Y，每个大小为1
            Y_t = torch.squeeze(Y_t, dim=0)  # (b, e)
            Ybar_t = torch.cat((Y_t, o_prev), dim=1)  # (b, e+h)
            dec_state, o_t, e_t = self.step(Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks)  # 计算
            combined_outputs.append(o_t)
            o_prev = o_t # 上一步输出的指针指向本次输出的 o_t
        combined_outputs = torch.stack(combined_outputs, dim=0)  # (b, h)->(tgt_len, b, h) 堆积成目标形状

        return combined_outputs

    def step(self, Ybar_t: torch.Tensor,
            dec_state: Tuple[torch.Tensor, torch.Tensor],
            enc_hiddens: torch.Tensor,
            enc_hiddens_proj: torch.Tensor,
            enc_masks: torch.Tensor) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        """ 计算 LSTM decoder 的每个前向步, 包括注意力计算.

        @param Ybar_t (Tensor): 连接好的张量  [Y_t o_prev], 形状 (b, e + h). decoder 的输入
                                b = batch size, e = embedding size, h = hidden size.
        @param dec_state (tuple(Tensor, Tensor)): 张量元组 形状都为 (b, h), b = batch size, h = hidden size.
                第一个张量是 decoder 的先前的隐藏状态, 第二个张量是 decoder 的先前的细胞状态.
        @param enc_hiddens (Tensor): Encoder 隐藏状态张量, 形状 (b, src_len, h * 2),
                                    b = batch size, src_len = 源的最大长度, h = hidden size.
        @param enc_hiddens_proj (Tensor): Encoder 隐藏状态张量, 从 (h * 2) 投影成 h. 张量形状 (b, src_len, h),
                                        b = batch size, src_len = 源的最大长度, h = hidden size.
        @param enc_masks (Tensor): 句子掩码张量，形状 (b, src_len),
                                    b = batch size, src_len = 源的最大长度.

        @returns dec_state (tuple (Tensor, Tensor)): 张量元组 两个张量形状都为 (b, h), b = batch size, h = hidden size.
                第一个张量是 decoder 的新隐藏状态, 第二个张量是 decoder 的新细胞状态.
        @returns combined_output (Tensor): 第t步连接的输出张量, 形状 (b, h), b = batch size, h = hidden size.
        @returns e_t (Tensor): 张量，形状 (b, src_len). 注意力分数分布.
                                这个函数之外不会使用到。
        """

        combined_output = None

        dec_state = self.decoder(Ybar_t,dec_state)
        dec_hidden, dec_cell = dec_state # 分开成两部分
        # (b, src_len) = (b, src_len, h)·(b, h) -> (b, src_len, 1) = (b, src_len, h)·(b, h, 1)
        e_t = torch.bmm(enc_hiddens_proj,torch.unsqueeze(dec_hidden,dim=2)) # (b, src_len, 1)
        e_t = torch.squeeze(e_t,dim=2) # (b, src_len)

        # 把掩码为1的e_t置为负无穷
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.bool(), -float('inf'))

        # 计算
        alpha_t = F.softmax(e_t,dim=1) # (b, src_len)
        alpha_t = torch.unsqueeze(alpha_t,dim=1) # (b, 1, src_len)
        a_t = torch.bmm(alpha_t,enc_hiddens) # (b, 1, h * 2) = (b, 1, src_len)·(b, src_len, h * 2)
        a_t = torch.squeeze(a_t,dim=1) # (b, 2h)
        U_t = torch.cat((a_t,dec_hidden),dim=1) # (b, 3h)
        V_t = self.combined_output_projection(U_t) # (b, h)
        O_t = self.dropout(torch.tanh(V_t)) # (b, h)

        combined_output = O_t
        return dec_state, combined_output, e_t


    def generate_sent_masks(self, enc_hiddens: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        """ 对 encoder 隐藏状态生成句子掩码

        @param enc_hiddens (Tensor): 需要编码的张量，形状 (b, src_len, 2*h), b = batch size,
                                     src_len = 源的最大长度, h = hidden size.
        @param source_lengths (List[int]): batch 中每个句子的实际长度.

        @returns enc_masks (Tensor): 句子掩码的张量，形状 (b, src_len),
                                    src_len = 源的最大长度, h = hidden size.
        """
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)  # 掩码矩阵
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1  # 超过句子长度的地方置为1
        return enc_masks.to(self.device)


    def beam_search(self, src_sent: List[str], beam_size: int=5, max_decoding_time_step: int=70) -> List[Hypothesis]:
        """ 给定单个源句子, 运行 beam search, 生成目标形式的结果.
        @param src_sent (List[str]): 一个源句子（词）
        @param beam_size (int): beam size，候选数
        @param max_decoding_time_step (int): 展开解码 RNN 的最大时间步
        @returns hypotheses (List[Hypothesis]): 假设列表, 每个假设有两个域:
                value: List[str]: 解码的目标句子, 用词序列表示
                score: float: 目标句子的对数似然
        """
        src_sents_var = self.vocab.src.to_input_tensor([src_sent], self.device)

        src_encodings, dec_init_vec = self.encode(src_sents_var, [len(src_sent)])
        src_encodings_att_linear = self.att_projection(src_encodings)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        eos_id = self.vocab.tgt['</s>']

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device) # (1,)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses) # 候选句数量

            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))

            y_tm1 = torch.tensor([self.vocab.tgt[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
            y_t_embed = self.model_embeddings.target(y_tm1)

            x = torch.cat([y_t_embed, att_tm1], dim=-1)
            # att_t形状(b, h)
            (h_t, cell_t), att_t, _  = self.step(x, h_tm1,
                                                      exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None)

            # 目标文字的概率对数
            log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1) # (词表长,)

            live_hyp_num = beam_size - len(completed_hypotheses) # beam_size - 完成的句子数
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1) # (候选句数,词表长)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num) # (候选句数,k)

            prev_hyp_ids = top_cand_hyp_pos / len(self.vocab.tgt) # 前序候选词id矩阵(候选句数,k)
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt) # 候选词id矩阵(候选句数,k)

            new_hypotheses = [] # 新候选句子
            live_hyp_ids = [] # 剩余候选id
            new_hyp_scores = [] # 新候选句分数
            # 按照k的数量迭代更新候选句
            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.tgt.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>': # 完成
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size: # 候选项达最大值
                break

            # 更新下一个状态
            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0: # 候选项为空
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True) # 得分降序

        return completed_hypotheses


    @property
    def device(self) -> torch.device:
        """ 决定使用CPU或GPU去放置张量.
        """
        return self.model_embeddings.source.weight.device


    @staticmethod
    def load(model_path: str):
        """ 从文件中加载模型.
        @param model_path (str): 模型路径
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NMT(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])

        return model


    def save(self, path: str):
        """ 保存模型到文件.
        @param path (str): 模型路径
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(embed_size=self.model_embeddings.embed_size, hidden_size=self.hidden_size, dropout_rate=self.dropout_rate),
            'vocab': self.vocab,
            'state_dict': self.state_dict() # 包含lr学习率
        }

        torch.save(params, path)