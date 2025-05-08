import os
import argparse
import json

import torch
torch.set_num_threads(4) 
from torch.optim import SGD, Adam
import copy

from pykt.models import train_model,evaluate,init_model,evaluate_only
from pykt.utils import debug_print,set_seed
from pykt.datasets import init_dataset4train
import datetime
import random
import numpy as np


def fix_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

fix_seed()

def get_device():
    if torch.backends.mps.is_available():  # Check for Apple Silicon GPU support
        return torch.device("mps")
    elif torch.cuda.is_available():  # Check for CUDA GPU support
        return torch.device("cuda")
    else:  # Fallback to CPU if neither MPS nor CUDA is available
        return torch.device("cpu")

#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device("cuda")
#os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:2'

def save_config(train_config, model_config, data_config, params, save_dir):
    d = {"train_config": train_config, 'model_config': model_config, "data_config": data_config, "params": params}
    save_path = os.path.join(save_dir, "config.json")
    with open(save_path, "w") as fout:
        json.dump(d, fout)

def main(params):
    print("HEYYYY The device is", device)
    # Some param initializations to ensure compatibility
    if "train_subset_rate" not in params:
        params["train_subset_rate"] = 1
    if "use_wandb" not in params:
        params['use_wandb'] = 1
    if "weighted_loss" not in params:
        print("As not specified, the weighted loss won't be applied")
        params["weighted_loss"] = 0 

    if params['use_wandb']==1:
        import wandb
        if "wandb_project_name" in params and params["wandb_project_name"] != "":
            wandb.init(project=params["wandb_project_name"])
        else:
            wandb.init()

    set_seed(params["seed"])
    model_name, dataset_name, fold, emb_type, save_dir = params["model_name"], params["dataset_name"], \
        params["fold"], params["emb_type"], params["save_dir"]
        
    debug_print(text = "load config files.",fuc_name="main")
    
    with open("../configs/kt_config.json") as f:
        config = json.load(f)
        train_config = config["train_config"]
        if model_name in ["dkvmn","deep_irt", "sakt", "saint","saint++", "akt", "atkt", "lpkt", "skvmn", "dimkt"]:
            train_config["batch_size"] = 64 ## because of OOM
        if model_name in ["simplekt", "bakt_time", "sparsekt"]:
            train_config["batch_size"] = 64 ## because of OOM
        if model_name in ["gkt"]:
            train_config["batch_size"] = 16 
        if model_name in ["qdkt","qikt"] and dataset_name in ['algebra2005','bridge2algebra2006']:
            train_config["batch_size"] = 32 
        model_config = copy.deepcopy(params)
        for key in ["model_name", "dataset_name", "emb_type", "save_dir", "fold", "seed"]:
            del model_config[key]
        # Emb_path should be read from data_config. 
        # data_config is later updated based on the params["emb_path"]. 
        if "emb_path" in model_config:
            del model_config["emb_path"]
        if 'batch_size' in params:
            train_config["batch_size"] = params['batch_size']
        if 'num_epochs' in params:
            train_config["num_epochs"] = params['num_epochs']
        # model_config = {"d_model": params["d_model"], "n_blocks": params["n_blocks"], "dropout": params["dropout"], "d_ff": params["d_ff"]}
    batch_size, num_epochs, optimizer = train_config["batch_size"], train_config["num_epochs"], train_config["optimizer"]

    with open("../configs/data_config.json") as fin:
        data_config = json.load(fin)
        # if emb_path is given, overwrite the path in data_config
        if "emb_path" in params and params["emb_path"] != "":
            data_config[dataset_name]["emb_path"] = params["emb_path"]
    if 'maxlen' in data_config[dataset_name]:#prefer to use the maxlen in data config
        train_config["seq_len"] = data_config[dataset_name]['maxlen']
    seq_len = train_config["seq_len"]

    print("Start init data")
    print(dataset_name, model_name, data_config[dataset_name], fold, batch_size)
    
    debug_print(text="init_dataset",fuc_name="main")
    if model_name not in ["dimkt"]:
        train_loader, valid_loader, *_ = init_dataset4train(dataset_name, model_name, data_config, fold, batch_size, train_subset_rate=params["train_subset_rate"])
    else:
        diff_level = params["difficult_levels"]
        train_loader, valid_loader, *_ = init_dataset4train(dataset_name, model_name, data_config, fold, batch_size, diff_level=diff_level, train_subset_rate=params["train_subset_rate"])

    params_str = "_".join([str(v) for k,v in params.items() if not k in ['other_config']])

    print(f"params: {params}, params_str: {params_str}")
    if params['add_uuid'] == 1 and params["use_wandb"] == 1:
        import uuid
        # if not model_name in ['saint','saint++']:
        #params_str = params_str+f"_{ str(uuid.uuid4())}"
        params_str = params_str
        folder_name = f"{str(uuid.uuid4())}"
    ckpt_path = os.path.join(save_dir, 'my_model')
    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)
    print(f"Start training model: {model_name}, embtype: {emb_type}, save_dir: {ckpt_path}, dataset_name: {dataset_name}")
    print(f"model_config: {model_config}")
    print(f"train_config: {train_config}")

    if model_name in ["dimkt"]:
        # del model_config['num_epochs']
        del model_config['weight_decay']

    save_config(train_config, model_config, data_config[dataset_name], params, ckpt_path)

    # Do the save for wandb
    # wandb.config.update(params)
    # wandb.config.update({"checkpoint_path": ckpt_path})

    learning_rate = params["learning_rate"]
    for remove_item in ['use_wandb','learning_rate','add_uuid','l2']:
        if remove_item in model_config:
            del model_config[remove_item]
    if model_name in ["saint","saint++", "sakt", "atdkt", "simplekt", "bakt_time", "sakt_que", "saint_que"]:
        model_config["seq_len"] = seq_len
        
    debug_print(text = "init_model",fuc_name="main")
    print(f"model_name:{model_name}")
    model = init_model(model_name, model_config, data_config[dataset_name], emb_type)
    print(f"model is {model}")
    if model_name == "hawkes":
        weight_p, bias_p = [], []
        for name, p in filter(lambda x: x[1].requires_grad, model.named_parameters()):
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        optdict = [{'params': weight_p}, {'params': bias_p, 'weight_decay': 0}]
        opt = torch.optim.Adam(optdict, lr=learning_rate, weight_decay=params['l2'])
    elif model_name == "iekt":
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    elif model_name == "dimkt":
        opt = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=params['weight_decay'])
    else:
        if optimizer == "sgd":
            opt = SGD(model.parameters(), learning_rate, momentum=0.9)
        elif optimizer == "adam":
            opt = Adam(model.parameters(), learning_rate)
   
    testauc, testacc = -1, -1
    window_testauc, window_testacc = -1, -1
    validauc, validacc = -1, -1
    best_epoch = -1
    save_model = True
    
    debug_print(text = "train model",fuc_name="main")
    
    if model_name == "rkt":
        dict_res = \
            train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, None, None, save_model, data_config[dataset_name], fold, use_wandb=params['use_wandb'], weighted_loss=params["weighted_loss"])
    else:
        # dict_res = train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, None, None, save_model, use_wandb=params['use_wandb'], weighted_loss=params["weighted_loss"])
        
        # ### For pretrained model and "evaluation only"
        from pykt.models import evaluate_only
        net = torch.load(os.path.join(ckpt_path, "current_model.ckpt"),  map_location=model.device)
        model.load_state_dict(net)
        
        # way 1
        # dict_res = evaluate_only.train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, None, None, save_model, use_wandb=params['use_wandb'], weighted_loss=params["weighted_loss"])
        
        # way 2
        # 1. 构造 questions 和 concepts
        questions = [
                [
                    "What is gravity?",
                    "Define photosynthesis.",
                    "How to solve a linear equation?",
                    "Explain Newton's third law.",
                    "What is the capital of France?",
                    "Describe the water cycle.",
                    "What is the Pythagorean theorem?",
                    "Explain how a plant makes food.",
                    "What is an atom?",
                    "Describe the process of mitosis."
                ],
                   [
                    "What is gravity?",
                    "Define photosynthesis.",
                    "How to solve a linear equation?",
                    "Explain Newton's third law.",
                    "What is the capital of France?",
                    "Describe the water cycle.",
                    "What is the Pythagorean theorem?",
                    "Explain how a plant makes food.",
                    "What is an atom?",
                    "Describe the process of mitosis."
                ],
                ]

        concepts = [
                   [
                    "physics; force",
                    "biology; plant",
                    "math; algebra",
                    "physics; mechanics",
                    "geography; capitals",
                    "science; environment",
                    "math; geometry",
                    "biology; photosynthesis",
                    "chemistry; matter",
                    "biology; cell division"
                ],
                   [
                    "physics; force",
                    "biology; plant",
                    "math; algebra",
                    "physics; mechanics",
                    "geography; capitals",
                    "science; environment",
                    "math; geometry",
                    "biology; photosynthesis",
                    "chemistry; matter",
                    "biology; cell division"
                ],
                ]

        # 2. （可选）如果你有已知的 ground truth
        labels = [[1, 1, 0, 1, 1, 0, 1, 0, 1, 1], [1, 1, 0, 1, 1, 0, 1, 0, 1, 1]]

        # 3. 将它们打包成 data 字典
        data_text = {
                    'questions': questions,
                    'concepts':  concepts,
                    # 'labels':    labels,    # 如果不传，内部会默认全 1
                }
        y = model.predict_one_step_g(data_text)
        print(y)
        input('')

        # questions = [
        # "What is the capital of France?",
        # "Explain how photosynthesis works.",
        # "Solve for x in the equation: 2x + 5 = 17."
        # ]
        # concepts = [
        #     "geography; capitals",
        #     "biology; plant physiology",
        #     "algebra; linear equations"
        # ]
        # # 端到端文本预测
        # probs, cog_ids, acq_ids, cog_desc, acq_desc = model.predict_text_sequence(
        #     questions, concepts, greedy=True
        # )
        # for i, q in enumerate(questions):
        #     print(f"Q{i+1}: {q}")
        #     print(f"  P(correct)      = {probs[i]:.3f}")
        #     print(f"  Cognition level = {cog_ids[i]} ({cog_desc[i]})")
        #     print(f"  Acquisition lvl = {acq_ids[i]} ({acq_desc[i]})")
        #     print()
        # dict_res = evaluate_only.train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, None, None, save_model, use_wandb=params['use_wandb'], weighted_loss=params["weighted_loss"])
       
    # if save_model:
    #     best_model = init_model(model_name, model_config, data_config[dataset_name], emb_type)
    #     net = torch.load(os.path.join(ckpt_path, emb_type+"_model_assisg2009.ckpt"))
    #     best_model.load_state_dict(net)

    print("fold\tmodelname\tembtype\ttestauc\ttestavgprc\ttestacc\twindow_testauc\twindow_testavgprc\twindow_testacc\tvalidauc\tvalidavgprc\tvalidacc\tbest_epoch")
    print(str(fold) + "\t" + model_name + "\t" + emb_type + "\t" + str(round(dict_res['test_auc'], 4)) + str(round(dict_res['test_avg_prc'], 4)) + "\t" + str(round(dict_res['test_acc'], 4)) + "\t" + str(round(dict_res['window_test_auc'], 4)) + str(round(dict_res['window_test_avg_prc'], 4)) + "\t" + str(round(dict_res['window_test_acc'], 4)) + "\t" + str(round(dict_res['valid_auc_checkpoint'], 4)) + str(round(dict_res['valid_avg_prc_checkpoint'], 4)) + "\t" + str(round(dict_res['valid_acc_checkpoint'], 4)) + "\t" + str(dict_res['best_epoch']))
    model_save_path = os.path.join(ckpt_path, emb_type+"_model.ckpt")
    print(f"end:{datetime.datetime.now()}")
    
    if params['use_wandb']==1:
        wandb.log({ 
                    "Final Validation AUC": dict_res['valid_auc_checkpoint'], 
                    "Final Validation AUPRC": dict_res['valid_avg_prc_checkpoint'], 
                    "Final Validation ACC": dict_res['valid_acc_checkpoint'],  
                    "best_epoch": dict_res['best_epoch'],
                    "model_save_path":model_save_path}, commit=True)
