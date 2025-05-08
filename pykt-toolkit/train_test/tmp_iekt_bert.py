#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整示例：  
1) 从 configs/data_config.json 自动加载 xes3g5m 超参  
2) 定义 IEKT+GPT 文本端到端推理  
3) 演示新文本批量预测
"""
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from transformers import BertTokenizer, BertModel

# 假设 pykt 工具箱在同级目录
from pykt.models.iekt_que import QueBaseModel        # 或者你的实际路径
from pykt.models.iekt_utils import mygru, funcs
from pykt.models.akt_que import QueEmbedder

class IEKTQueNet(nn.Module):
    def __init__(self,
        num_q, num_c, emb_size, max_concepts,
        lamb=40, n_layer=1, cog_levels=10, acq_levels=10,
        dropout=0, gamma=0.93, emb_path="", flag_load_emb=False,
        flag_emb_freezed=False, device='cpu'
    ):
        super().__init__()
        self.emb_size   = emb_size
        self.gamma      = gamma
        self.cog_levels = cog_levels
        self.acq_levels = acq_levels
        # policy heads
        self.select_preemb = funcs(n_layer, emb_size*3, cog_levels, dropout)
        self.predictor     = funcs(n_layer, emb_size*5, 1, dropout)
        self.checker_emb   = funcs(n_layer, emb_size*12, acq_levels, dropout)
        # cognition/acquisition embeddings
        self.cog_matrix = nn.Parameter(torch.randn(cog_levels, emb_size*2).to(device))
        self.acq_matrix = nn.Parameter(torch.randn(acq_levels, emb_size*2).to(device))
        # state update GRU
        self.gru_h = mygru(0, emb_size*4, emb_size)
        # fallback pure-ID embedder（不用于 GPT 文本推理）
        self.que_emb = QueEmbedder(num_q, emb_size*2, emb_path,
                                   flag_load_emb, flag_emb_freezed, "iekt_que")
        self.sigmoid = nn.Sigmoid()

    def pi_cog_func(self, x):
        return F.softmax(self.select_preemb(x), dim=1)

    def pi_sens_func(self, x):
        return F.softmax(self.checker_emb(x), dim=1)

    def obtain_v(self, h, x_prev, emb, v_override=None):
        # v_override: [B, emb_size*2] 直接用上游文本编码结果
        if v_override is None:
            raise RuntimeError("GPT 推理需传入 v_override")
        v = v_override
        h_v = torch.cat([h, v], dim=1)
        logits = self.predictor(torch.cat([h_v, emb], dim=1))
        return h_v, v, logits, x_prev

    def update_state(self, h, v, emb, operate):
        v_cat = torch.cat([v * operate,       v * (1-operate)], dim=1)
        e_cat = torch.cat([emb * (1-operate), emb * operate   ], dim=1)
        return self.gru_h(v_cat + e_cat, h)


class IEKTQue(QueBaseModel):
    def __init__(self,
        num_q, num_c, emb_size, max_concepts,
        lamb=40, n_layer=1, cog_levels=10, acq_levels=10,
        dropout=0, gamma=0.93,
        emb_type='qid', emb_path="",
        flag_load_emb=False, flag_emb_freezed=False,
        pretrain_dim=768, device='cpu', seed=0
    ):
        super().__init__(model_name="iekt_que",
                         emb_type=emb_type,
                         emb_path=emb_path,
                         pretrain_dim=pretrain_dim,
                         device=device, seed=seed)
        self.model = IEKTQueNet(
            num_q, num_c, emb_size, max_concepts,
            lamb, n_layer, cog_levels, acq_levels,
            dropout, gamma, emb_path,
            flag_load_emb, flag_emb_freezed, device
        ).to(device)

        # 冻结 BERT 并线性投影到 emb_size
        BERT_PATH = './bert-tiny'
        self.bert_tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
        self.bert_model     = BertModel    .from_pretrained(BERT_PATH).to(device)
        for p in self.bert_model.parameters():
            p.requires_grad = False
        self.projection = nn.Linear(pretrain_dim, emb_size).to(device)

    @torch.no_grad()
    def embed_text(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        toks = self.bert_tokenizer(
            texts, padding=True, truncation=True,
            return_tensors='pt'
        ).to(self.device)
        out = self.bert_model(**toks)
        # 取 [CLS] 向量 并投影
        cls_vec = out.last_hidden_state[:,0,:]  # [B,768]
        return self.projection(cls_vec)         # [B,emb_size]

    @torch.no_grad()
    def predict_gpt_sequence(self, questions, concepts, greedy=True):
        """
        输入:
          questions: List[str]
          concepts : List[str]
        输出:
          probs: List[float]
          cog_ids, acq_ids: List[int]
          cog_desc, acq_desc: List[str]
        """
        # 文本编码
        q_emb = self.embed_text(questions)   # [N,64]
        c_emb = self.embed_text(concepts)    # [N,64]

        N = q_emb.size(0)
        # 初始化
        h      = torch.zeros(1, self.model.emb_size, device=self.device)
        x_prev = torch.zeros(1, 1, self.model.emb_size*2, device=self.device)
        probs, cog_ids, acq_ids, cog_desc, acq_desc = [], [], [], [], []

        def desc(idx, total):
            r = idx/float(total-1)
            return "low"    if r<0.33 else \
                   "medium" if r<0.66 else \
                   "high"

        for i in range(N):
            # 把 question+concept 拼成 [1,emb_size*2]
            v_t = torch.cat([q_emb[i:i+1], c_emb[i:i+1]], dim=1)
            # cognition 分布
            ques_h = torch.cat([v_t, h], dim=1)
            pi_cog = self.model.pi_cog_func(ques_h)
            idx_cog = pi_cog.argmax(dim=1) if greedy else Categorical(pi_cog).sample()
            emb_p   = self.model.cog_matrix[idx_cog]  # [1,emb_size*2]

            # 预测 & record
            h_v, v, logits, _ = self.model.obtain_v(h, x_prev, emb_p, v_override=v_t)
            prob = torch.sigmoid(logits).item()
            probs.append(prob)
            cid = idx_cog.item()
            cog_ids.append(cid)
            cog_desc.append(desc(cid, self.model.cog_levels))

            # acquisition
            lbl = torch.tensor([[1.0 if prob>0.5 else 0.0]],
                               device=self.device)
            out_x = torch.cat([h_v*lbl, h_v*(1-lbl)], dim=1)
            pi_sens = self.model.pi_sens_func(out_x)
            idx_sens = pi_sens.argmax(dim=1) if greedy else Categorical(pi_sens).sample()
            aid = idx_sens.item()
            acq_ids.append(aid)
            acq_desc.append(desc(aid, self.model.acq_levels))

            # 更新 hidden state
            h = self.model.update_state(h, v, self.model.acq_matrix[idx_sens], lbl)

        return probs, cog_ids, acq_ids, cog_desc, acq_desc


def load_model_from_ckpt(ckpt_path, cfg, device):
    """
    根据 data_config.json 的 cfg 自动构造 IEKTQue
    """
    model = IEKTQue(
        num_q        = cfg["num_q"],
        num_c        = cfg["num_c"],
        emb_size     = 64,
        max_concepts = cfg["max_concepts"],
        device       = device
    )
    sd = torch.load(ckpt_path, map_location=device)
    model.model.load_state_dict(sd)
    model.eval()
    return model


if __name__ == "__main__":
    # 1) 读取 xes3g5m 超参
    with open("../configs/data_config.json") as f:
        data_config = json.load(f)
    cfg = data_config["xes3g5m"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from pykt.models import train_model,evaluate,init_model,evaluate_only

    model = init_model(model_name='iekt', model_config=data_config, data_config=cfg, emb_type='qid')
    model  = load_model_from_ckpt("saved_model/current_model.ckpt", cfg, device)

    # 2) 新文本示例
    questions = [
        "Explain the Pythagorean theorem.",
        "What is photosynthesis?",
        "Solve for y: 3y - 4 = 11."
    ]
    concepts = [
        "geometry; right triangle",
        "biology; plant physiology",
        "algebra; linear equations"
    ]

    # 3) 端到端预测
    probs, cog_ids, acq_ids, cog_desc, acq_desc = model.predict_gpt_sequence(
        questions, concepts, greedy=True
    )

    # 4) 打印
    for i, q in enumerate(questions):
        print(f"Q{i+1}: {q}")
        print(f"  P(correct)      = {probs[i]:.3f}")
        print(f"  Cognition level = {cog_ids[i]} ({cog_desc[i]})")
        print(f"  Acquisition lvl = {acq_ids[i]} ({acq_desc[i]})")
        print()
