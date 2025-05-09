import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy 
from .que_base_model import QueBaseModel,QueEmb
from torch.distributions import Categorical
from .iekt_utils import mygru,funcs
from pykt.utils import debug_print
from .akt_que import QueEmbedder
from transformers import BertTokenizer, BertModel

class IEKTQueNet(nn.Module): 
    def __init__(self, num_q,num_c,emb_size,max_concepts,lamb=40,n_layer=1,cog_levels=10,acq_levels=10,dropout=0,gamma=0.93, emb_type='qc_merge', emb_path="", flag_load_emb=False, flag_emb_freezed=False, pretrain_dim=768,device='cpu'):
        super().__init__()
        self.model_name = "iekt_que"
        debug_print(f"The flag_load_emb is {flag_load_emb} in IEKTQueNet.",fuc_name=self.model_name)
        self.emb_size = emb_size
        self.concept_num = num_c
        self.max_concept = max_concepts
        self.device = device
        self.emb_type = emb_type
        self.predictor = funcs(n_layer, emb_size * 5, 1, dropout)
        self.cog_matrix = nn.Parameter(torch.randn(cog_levels, emb_size * 2).to(self.device), requires_grad=True) 
        self.acq_matrix = nn.Parameter(torch.randn(acq_levels, emb_size * 2).to(self.device), requires_grad=True)
        self.select_preemb = funcs(n_layer, emb_size * 3, cog_levels, dropout)#MLP
        self.checker_emb = funcs(n_layer, emb_size * 12, acq_levels, dropout) 
        self.prob_emb = nn.Parameter(torch.randn(num_q, emb_size).to(self.device), requires_grad=True)#题目表征
        self.gamma = gamma
        self.lamb = lamb
        self.gru_h = mygru(0, emb_size * 4, emb_size)
        self.concept_emb = nn.Parameter(torch.randn(self.concept_num, emb_size).to(self.device), requires_grad=True)#知识点表征
        self.sigmoid = torch.nn.Sigmoid()
        # self.que_emb = QueEmb(num_q=num_q,num_c=num_c,emb_size=emb_size,emb_type=self.emb_type,model_name=self.model_name,device=device,
        #                      emb_path=emb_path,pretrain_dim=pretrain_dim)
        
        # NOTE: QueEmbedder has emb_size*2. The reason is, original implementation was concat'in 2 embs (que and concept)
        # We had to mimic this behaviour to match the dimensionality.
        self.que_emb = QueEmbedder(num_q, emb_size*2, emb_path, flag_load_emb, flag_emb_freezed, self.model_name)


    def get_ques_representation(self, q, c):
        """Get question representation equation 3

        Args:
            q (_type_): question ids
            c (_type_): concept ids -> DEPRECATED, not use but signature kept

        Returns:
            _type_: _description_
        """
       
        v = self.que_emb(q)

        return v


    def pi_cog_func(self, x, softmax_dim = 1):
        return F.softmax(self.select_preemb(x), dim = softmax_dim)

    def obtain_v(self, q, c, h, x, emb):
        """_summary_

        Args:
            q (_type_): _description_
            c (_type_): _description_
            h (_type_): _description_
            x (_type_): _description_
            emb (_type_): m_t

        Returns:
            _type_: _description_
        """

        #debug_print("start",fuc_name='obtain_v')
        v = self.get_ques_representation(q,c)
        predict_x = torch.cat([h, v], dim = 1)#equation4
        h_v = torch.cat([h, v], dim = 1)#equation4 为啥要计算两次？
        prob = self.predictor(torch.cat([
            predict_x, emb
        ], dim = 1))#equation7
        return h_v, v, prob, x

    def update_state(self, h, v, emb, operate):
        """_summary_

        Args:
            h (_type_): rnn的h
            v (_type_): question 表示
            emb (_type_): s_t knowledge acquistion sensitivity
            operate (_type_): label

        Returns:
            next_p_state {}: _description_
        """
        #equation 13
        v_cat = torch.cat([
            v.mul(operate.repeat(1, self.emb_size * 2)),
            v.mul((1 - operate).repeat(1, self.emb_size * 2))], dim = 1)#v_t扩展，分别对应正确的错误的情况
        e_cat = torch.cat([
            emb.mul((1-operate).repeat(1, self.emb_size * 2)),
            emb.mul((operate).repeat(1, self.emb_size * 2))], dim = 1)# s_t 扩展，分别对应正确的错误的情况
        inputs = v_cat + e_cat#起到concat作用
        
        h_t_next = self.gru_h(inputs, h)#equation14
        return h_t_next
    

    def pi_sens_func(self, x, softmax_dim = 1):
        return F.softmax(self.checker_emb(x), dim = softmax_dim)
    
    



class IEKTQue(QueBaseModel):
    def __init__(self, num_q,num_c,emb_size,max_concepts,lamb=40,n_layer=1,cog_levels=10,acq_levels=10,dropout=0,gamma=0.93, emb_type='qid', emb_path="", flag_load_emb=False, flag_emb_freezed=False, pretrain_dim=768,device='cpu',seed=0, **kwargs):
        model_name = "iekt"
        super().__init__(model_name=model_name,emb_type=emb_type,emb_path=emb_path,pretrain_dim=pretrain_dim,device=device,seed=seed)

        self.model = IEKTQueNet(num_q=num_q,num_c=num_c,lamb=lamb,emb_size=emb_size,max_concepts=max_concepts,n_layer=n_layer,cog_levels=cog_levels,acq_levels=acq_levels,dropout=dropout,gamma=gamma, emb_type=emb_type, emb_path=emb_path, flag_load_emb=flag_load_emb, flag_emb_freezed=flag_emb_freezed, pretrain_dim=pretrain_dim,device=device)

        self.model = self.model.to(device)
        self.loss_func = self._get_loss_func("binary_crossentropy")
        # self.step = 0
                
        # BERT for text embed
        BERT_PATH = './bert-tiny'
        self.bert_tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
        self.bert_model     = BertModel.from_pretrained(BERT_PATH).to(device)
        for param in self.bert_model.parameters():
            param.requires_grad = False
        BERT_DIM = 128
        # divide by 2 => one for question embedding, and one for concept embedding
        self.projection_layer = nn.Linear(BERT_DIM, 64).to(device)
    
    def train_one_step(self,data,process=True,weighted_loss=0):
        # self.step+=1
        # debug_print(f"step is {self.step},data is {data}","train_one_step")
        # debug_print(f"step is {self.step}","train_one_step")
        # YO Removal BCELoss = torch.nn.BCEWithLogitsLoss()
        
        data_new,emb_action_list,p_action_list,states_list,pre_state_list,reward_list,predict_list,ground_truth_list = self.predict_one_step(data,return_details=True,process=process)
        data_len = data_new['cc'].shape[0]
        seq_len = data_new['cc'].shape[1]

        #以下是强化学习部分内容
        seq_num = torch.where(data['qseqs']!=0,1,0).sum(axis=-1)+1
        emb_action_tensor = torch.stack(emb_action_list, dim = 1)
        p_action_tensor = torch.stack(p_action_list, dim = 1)
        state_tensor = torch.stack(states_list, dim = 1)
        pre_state_tensor = torch.stack(pre_state_list, dim = 1)
        reward_tensor = torch.stack(reward_list, dim = 1).float() / (seq_num.unsqueeze(-1).repeat(1, seq_len)).float()#equation15
        logits_tensor = torch.stack(predict_list, dim = 1)
        ground_truth_tensor = torch.stack(ground_truth_list, dim = 1)
        loss = []
        tracat_logits = []
        tracat_ground_truth = []
        
        for i in range(0, data_len):
            # print(i)
            this_seq_len = seq_num[i]
            this_reward_list = reward_tensor[i]
            this_cog_state = torch.cat([pre_state_tensor[i][0: this_seq_len],
                                    torch.zeros(1, pre_state_tensor[i][0].size()[0]).to(self.device)
                                    ], dim = 0)
            this_sens_state = torch.cat([state_tensor[i][0: this_seq_len],
                                    torch.zeros(1, state_tensor[i][0].size()[0]).to(self.device)
                                    ], dim = 0)

            td_target_cog = this_reward_list[0: this_seq_len].unsqueeze(1)
            delta_cog = td_target_cog
            delta_cog = delta_cog.detach().cpu().numpy()

            td_target_sens = this_reward_list[0: this_seq_len].unsqueeze(1)
            delta_sens = td_target_sens
            delta_sens = delta_sens.detach().cpu().numpy()

            advantage_lst_cog = []
            advantage = 0.0
            for delta_t in delta_cog[::-1]:
                advantage = self.model.gamma * advantage + delta_t[0]#equation17
                advantage_lst_cog.append([advantage])
            advantage_lst_cog.reverse()
            advantage_cog = torch.tensor(advantage_lst_cog, dtype=torch.float).to(self.device)
            
            pi_cog = self.model.pi_cog_func(this_cog_state[:-1])
            # Below line gives error when i == 57 . It is MPS Assertion error.
            pi_a_cog = pi_cog.gather(1,p_action_tensor[i][0: this_seq_len].unsqueeze(1))

            loss_cog = -torch.log(pi_a_cog) * advantage_cog#equation16
            
            loss.append(torch.sum(loss_cog))

            advantage_lst_sens = []
            advantage = 0.0
            for delta_t in delta_sens[::-1]:
                # advantage = args.gamma * args.beta * advantage + delta_t[0]
                advantage = self.model.gamma * advantage + delta_t[0]
                advantage_lst_sens.append([advantage])
            advantage_lst_sens.reverse()
            advantage_sens = torch.tensor(advantage_lst_sens, dtype=torch.float).to(self.device)
            
            pi_sens = self.model.pi_sens_func(this_sens_state[:-1])
            pi_a_sens = pi_sens.gather(1,emb_action_tensor[i][0: this_seq_len].unsqueeze(1))

            loss_sens = - torch.log(pi_a_sens) * advantage_sens#equation18
            loss.append(torch.sum(loss_sens))
            

            this_prob = logits_tensor[i][0: this_seq_len]
            this_groud_truth = ground_truth_tensor[i][0: this_seq_len]

            tracat_logits.append(this_prob)
            tracat_ground_truth.append(this_groud_truth)

        # YO Removal bce = BCELoss(torch.cat(tracat_logits, dim = 0), torch.cat(tracat_ground_truth, dim = 0))   
        y_pred = torch.sigmoid(torch.cat(tracat_logits, dim = 0))
        y_true = torch.cat(tracat_ground_truth, dim = 0)
        y_mask = torch.ones_like(y_true) == 1
        bce = self.get_loss(y_pred,y_true,y_mask, weighted_loss=weighted_loss)

        y = torch.cat(tracat_logits, dim = 0)
        label_len = torch.cat(tracat_ground_truth, dim = 0).size()[0]
        loss_l = sum(loss)
        loss = self.model.lamb * (loss_l / label_len) +  bce#equation21
        # YO Removla return y,loss
        return y_pred,loss

    def predict_one_step(self,data,return_details=False,process=True):
        sigmoid_func = torch.nn.Sigmoid()
        data_new = self.batch_to_device(data,process)

        # device = torch.device('cuda')
        # T = 200  # 注意：设置为200，去掉一个后变成199
        # data_new = {
        #     'cq': torch.randint(1, 100, (1, T), dtype=torch.long, device=device),
        #     'cc': torch.randint(1, 50, (1, T), dtype=torch.long, device=device),
        #     'cr': torch.randint(0, 2, (1, T), dtype=torch.long, device=device),
        #     'qseqs': torch.randint(1, 100, (1, T), dtype=torch.long, device=device)
        # }


        data_len = data_new['cc'].shape[0]
        seq_len = data_new['cc'].shape[1]
        h = torch.zeros(data_len, self.model.emb_size).to(self.device)
        batch_probs, uni_prob_list, actual_label_list, states_list, reward_list =[], [], [], [], []
        p_action_list, pre_state_list, emb_action_list, op_action_list, actual_label_list, predict_list, ground_truth_list = [], [], [], [], [], [], []

        rt_x = torch.zeros(data_len, 1, self.model.emb_size * 2).to(self.device)
        for seqi in range(0, seq_len):#序列长度
            #debug_print(f"start data_new, c is {data_new}",fuc_name='train_one_step')
            ques_h = torch.cat([
                self.model.get_ques_representation(q=data_new['cq'][:,seqi], c=data_new['cc'][:,seqi]),
                h], dim = 1)#equation4
            # d = 64*3 [题目,知识点,h]
            # print('ques_h', ques_h.shape)
            flip_prob_emb = self.model.pi_cog_func(ques_h)

            m = Categorical(flip_prob_emb)#equation 5 的 f_p
            emb_ap = m.sample()#equation 5
            emb_p = self.model.cog_matrix[emb_ap,:]#equation 6

            h_v, v, logits, rt_x = self.model.obtain_v(q=data_new['cq'][:,seqi], c=data_new['cc'][:,seqi], 
                                                        h=h, x=rt_x, emb=emb_p)#equation 7
            prob = sigmoid_func(logits)#equation 7 sigmoid

            out_operate_groundtruth = data_new['cr'][:,seqi].unsqueeze(-1) #获取标签
            
            out_x_groundtruth = torch.cat([
                h_v.mul(out_operate_groundtruth.repeat(1, h_v.size()[-1]).float()),
                h_v.mul((1-out_operate_groundtruth).repeat(1, h_v.size()[-1]).float())],
                dim = 1)#equation9

            out_operate_logits = torch.where(prob > 0.5, torch.tensor(1).to(self.device), torch.tensor(0).to(self.device)) 
            out_x_logits = torch.cat([
                h_v.mul(out_operate_logits.repeat(1, h_v.size()[-1]).float()),
                h_v.mul((1-out_operate_logits).repeat(1, h_v.size()[-1]).float())],
                dim = 1)#equation10                
            out_x = torch.cat([out_x_groundtruth, out_x_logits], dim = 1)#equation11
            # print(f"data_new['cr'] is {data_new['cr']}")
            ground_truth = data_new['cr'][:,seqi]
            # print(f"ground_truth shape is {ground_truth.shape},ground_truth is {ground_truth}")
            flip_prob_emb = self.model.pi_sens_func(out_x)##equation12中的f_e

            m = Categorical(flip_prob_emb)
            emb_a = m.sample()
            emb = self.model.acq_matrix[emb_a,:]#equation12 s_t
            # print(f"emb_a shape is {emb_a.shape}")
            # print(f"emb shape is {emb.shape}")
            
            h = self.model.update_state(h, v, emb, ground_truth.unsqueeze(1))#equation13～14
           
            uni_prob_list.append(prob.detach())
            
            emb_action_list.append(emb_a)#s_t 列表
            p_action_list.append(emb_ap)#m_t
            states_list.append(out_x)
            pre_state_list.append(ques_h)#上一个题目的状态
            
            ground_truth_list.append(ground_truth)
            predict_list.append(logits.squeeze(1))
            this_reward = torch.where(out_operate_logits.squeeze(1).float() == ground_truth,
                            torch.tensor(1).to(self.device), 
                            torch.tensor(0).to(self.device))# if condition x else y,这里相当于统计了正确的数量
            reward_list.append(this_reward)
        prob_tensor = torch.cat(uni_prob_list, dim = 1)
        if return_details:
            return data_new,emb_action_list,p_action_list,states_list,pre_state_list,reward_list,predict_list,ground_truth_list
        else:
            return prob_tensor[:,1:]
    
    @torch.no_grad()
    def embed_text(self, texts):
        if isinstance(texts, str): texts = [texts]
        toks = self.bert_tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
        out = self.bert_model(**toks)
        return out.last_hidden_state[:,0,:]
    
    def predict_one_step_g(self, data, return_details=False, process=True):
        sigmoid_func = torch.nn.Sigmoid()

        # —— 1) 如果传入的是 {'questions':…, 'concepts':…, 'labels':…} 模式 —— #
        if isinstance(data, dict) and 'questions' in data and 'concepts' in data:
            questions = data['questions']
            concepts  = data['concepts']
            labels    = data.get('labels', None)

            B = len(questions)
            T = len(questions[0])
            if labels is None:
                labels = [[1]*T for _ in range(B)]
            labels_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)

            # BERT flatten 编码
            flat_q = [q for seq in questions for q in seq]
            flat_c = [c for seq in concepts for c in seq]
            toks_q = self.bert_tokenizer(flat_q, padding=True, truncation=True, return_tensors='pt')\
                            .to(self.device)
            toks_c = self.bert_tokenizer(flat_c, padding=True, truncation=True, return_tensors='pt')\
                            .to(self.device)
            out_q = self.bert_model(**toks_q).last_hidden_state[:,0,:]  # [B*T, H]
            out_c = self.bert_model(**toks_c).last_hidden_state[:,0,:]
            vec_q = self.projection_layer(out_q)  # [B*T, D]
            vec_c = self.projection_layer(out_c)
            q_c_vec = torch.cat([vec_q, vec_c], dim=-1).view(B, T, -1)

            # 构造一个 “伪 batch” 给后面老逻辑使用
            zero_ids = torch.zeros(B, T, dtype=torch.long, device=self.device)
            data_new = {
                'cq':      zero_ids,      # 占位
                'cc':      zero_ids,
                'cr':      labels_tensor, # ground-truth
                'qseqs':   torch.ones_like(zero_ids),
                'q_c_vec': q_c_vec,       # 打标量分支
            }

        else:
            # —— 2) 普通 PyKT batch —— #
            data_new = self.batch_to_device(data, process)

        # —— 接下来完全沿用原有 predict_one_step，只有两处“分支” —— #
        B, T = data_new['cr'].shape
        h    = torch.zeros(B, self.model.emb_size, device=self.device)
        rt_x = torch.zeros(B, 1, self.model.emb_size*2, device=self.device)

        uni_prob_list = []
        emb_action_list = []
        p_action_list   = []
        states_list     = []
        pre_state_list  = []
        predict_list    = []
        ground_truth_list = []
        reward_list       = []

        # 保存原方法，后面 vector 分支会临时替换它
        orig_get_repr = self.model.get_ques_representation

        for t in range(T):
            # —— 判断向量分支 or ID 分支 —— #
            if 'q_c_vec' in data_new:
                v_input = data_new['q_c_vec'][:, t]   # [B, 2D]
                # 临时让 get_ques_representation 返回 v_input
                self.model.get_ques_representation = lambda q, c, _v=v_input: _v
                use_vec = True
            else:
                v_input = None
                use_vec = False

            # 拼状态做 CE
            ques_h = torch.cat([
                self.model.get_ques_representation(
                    q=data_new['cq'][:,t], c=data_new['cc'][:,t]
                ), h], dim=1)
            prob_cog = self.model.pi_cog_func(ques_h)
            m_cog    = Categorical(prob_cog)
            idx_cog  = m_cog.sample()
            emb_p    = self.model.cog_matrix[idx_cog, :]

            # 恢复原 repr 函数，后续不会重复覆盖
            if use_vec:
                self.model.get_ques_representation = orig_get_repr

            # 主干推理
            if use_vec:
                # 此时 obtain_v 会内部调用我们刚才恢复后的 get_ques_representation
                h_v, v, logits, rt_x = self.model.obtain_v(
                    q=data_new['cq'][:,t],
                    c=data_new['cc'][:,t],
                    h=h, x=rt_x, emb=emb_p
                )
            else:
                h_v, v, logits, rt_x = self.model.obtain_v(
                    q=data_new['cq'][:,t],
                    c=data_new['cc'][:,t],
                    h=h, x=rt_x, emb=emb_p
                )

            prob = sigmoid_func(logits)

            # 构造 out_x
            gt   = data_new['cr'][:,t]
            ogt  = gt.unsqueeze(-1).float()
            x_gt = torch.cat([ h_v * ogt.repeat(1,h_v.size(-1)),
                               h_v * (1-ogt).repeat(1,h_v.size(-1)) ], dim=1)
            opl = (prob>0.5).float()
            x_pr= torch.cat([ h_v * opl.repeat(1,h_v.size(-1)),
                              h_v * (1-opl).repeat(1,h_v.size(-1)) ], dim=1)
            out_x = torch.cat([x_gt, x_pr], dim=1)

            # KASE
            prob_sens = self.model.pi_sens_func(out_x)
            m_sens    = Categorical(prob_sens)
            idx_sens  = m_sens.sample()
            emb_sens  = self.model.acq_matrix[idx_sens, :]

            # 更新状态
            h = self.model.update_state(h, v, emb_sens, gt.unsqueeze(1))

            # 保存结果
            uni_prob_list.append(prob.detach())
            emb_action_list.append(idx_sens)
            p_action_list.append(idx_cog)
            states_list.append(out_x)
            pre_state_list.append(ques_h)
            predict_list.append(logits.squeeze(1))
            ground_truth_list.append(gt)
            reward_list.append((opl.squeeze(1)==gt).float())

        # 还原 get_ques_representation
        self.model.get_ques_representation = orig_get_repr

        probs = torch.cat(uni_prob_list, dim=1)  # [B, T]

        if return_details:
            return (data_new,
                    emb_action_list, p_action_list,
                    states_list, pre_state_list,
                    reward_list,
                    predict_list, ground_truth_list)
        else:
            # 按原来习惯，丢掉第一个 warmup 步
            return probs[:, 1:]

