from transformers import BertModel, BertTokenizer, BertConfig
import torch
import json
from tqdm import tqdm
from torch.nn.functional import cosine_similarity
import os
import numpy as np
from numpy import linalg
from sklearn.preprocessing import normalize

# cd "representation_learning" folder

# Get the folder to load the trained model 
folder_saved_model = '../pykt-toolkit/train_test/bert-tiny/' # Experiment folder

path_data_questions = '../data/XES3G5M/metadata/questions_translated_kc_sol_annotated_mapped.json'
path_kc_questions_map = '../data/XES3G5M/metadata/kc_questions_map.json'

with open(path_data_questions, 'r') as file:
    data_questions = json.load(file)

with open(path_kc_questions_map, 'r') as file:
    kc_questions_map = json.load(file)

embeddings_save_folder = "../data/XES3G5M/metadata/embeddings/representation_learning"

if not os.path.exists(embeddings_save_folder):
    os.makedirs(embeddings_save_folder)

BERT_PATH = '../pykt-toolkit/train_test/bert-tiny'

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

# Create a configuration object or load it if you have saved one
config = BertConfig.from_pretrained(BERT_PATH)

# Initialize the model with this configuration
model = BertModel.from_pretrained(BERT_PATH)
for param in model.parameters():
    param.requires_grad = False

# Adjust the model's token embeddings to account for new tokens before loading the weights
model.resize_token_embeddings(len(tokenizer))

# # Load the model weights
# model.load_state_dict(torch.load(folder_saved_model + '/bert_finetuned.bin'))



# Move the model to the appropriate computing device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Set the model to training or evaluation mode as needed
model = model.eval()  # or model.train() if you continue training
# Constants
BATCH_SIZE = 1024  # Define your batch size here

# Helper function to batch text data and convert to embeddings
def text_to_embeddings(texts, max_length=128):
    embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Generating Embeddings"):
        batch_texts = texts[i:i + BATCH_SIZE]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state[:, 0, :])  # Extract [CLS] token embeddings
    return torch.cat(embeddings, dim=0)
list_questions = [value['question'] for key, value in data_questions.items()]
list_sol_steps = [[sol for sol in value['step_by_step_solution_list']] for key,value in data_questions.items()]

#Prepend special tokens 
questions = ['[Q] ' + q for q in list_questions]
sol_steps = [['[S] ' + step for step in sol_steps] for sol_steps in list_sol_steps]
#Get the embeddings
question_embeddings = text_to_embeddings(questions)

# Flatten the solution steps and prepend with special token
flat_solution_steps = [step for sublist in sol_steps for step in sublist]
flat_solution_embeddings = text_to_embeddings(flat_solution_steps)

# Map flat embeddings back to their respective lists using original lengths
sol_step_embeddings = []
start_idx = 0
for steps in sol_steps:
    end_idx = start_idx + len(steps)
    sol_step_embeddings.append(flat_solution_embeddings[start_idx:end_idx])
    start_idx = end_idx
# Convert these embeddings to numpy array or lists to have necessary pre-computations
np_question_embeddings = question_embeddings.cpu().detach().numpy()

np_sol_step_embeddings = []
for i in range(len(sol_step_embeddings)):
    np_sol_step_embeddings.append(sol_step_embeddings[i].cpu().detach().numpy())

np_sol_step_embeddings_mean = []
for i in range(len(np_sol_step_embeddings)):
    np_sol_step_embeddings_mean.append(np_sol_step_embeddings[i].mean(axis=0))
dict_emb = {}
for i in range(len(np_question_embeddings)):
    emb_q = np_question_embeddings[i].copy().reshape(1,-1)
    emb_sol = np_sol_step_embeddings_mean[i].copy().reshape(1,-1)
    emb = (emb_q + emb_sol)/2

    norm_emb = normalize(emb, axis=1, norm='l2').flatten()
    dict_emb[str(i)] = norm_emb.tolist()

save_path = os.path.join(embeddings_save_folder, 'qid2content_sol_avg_emb.json')

with open(save_path, 'w') as f:
    json.dump(dict_emb, f)