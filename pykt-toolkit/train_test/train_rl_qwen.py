import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer # Ensure these are imported

# ... (your existing imports and set_seed function)


model_name = "C:/Users/zhaoc/Desktop/KCQRL-main/pykt-toolkit/pykt/models/Qwen3-0.6B"
device = 'cuda'

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Set pad_token_id for the tokenizer if it's not already set
# if tokenizer.pad_token_id is None:
#     tokenizer.pad_token_id = tokenizer.eos_token_id # Or another appropriate padding token

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# Ensure tokenizer pad_token_id is set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 2. 创建假数据集
class FakeDataset(Dataset):
    def __init__(self, size=10):
        self.data = [f"This morning I went to the {random.choice(['park', 'classroom', 'car', 'moon'])}."
                     for _ in range(size)]

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

dataset = FakeDataset(size=4) # Your dataset has 4 items, which matches your batch_size

# 3. 初始化 PPO 训练器
ppo_config = {
    "learning_rate": 1.41e-5,
    "steps": 1,
    "mini_batch_size": 1, # Assuming your training batch size is 4
    "batch_size": 1,      # This is the total number of samples processed per PPO step
    "target_kl": 0.1,
    "ppo_epochs": 4,
}
config = PPOConfig(**ppo_config)

ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset)

generation_kwargs = {
    "do_sample": False,
    "max_new_tokens": 300,
    "num_beams": 8,
    "num_return_sequences": 1, # Keep this at 1 for simplicity per prompt
    "length_penalty": -1.0,
    "no_repeat_ngram_size": 2,
    "eos_token_id": tokenizer.eos_token_id,
    "pad_token_id": tokenizer.pad_token_id,
    "early_stopping": True
}

print("Testing model before RL training...")

test_prompts = []
for _ in range(config.batch_size):
    m = "Rewrite the following instruction via rephrasing and/or adding specific requirements. Use illustrative description if needed. Output the new instruction only."
    t = "Write an instruction that guides an LLM to generate a practice exercise that includes the knowledge concepts of 'Understanding multiplication as repeated addition' and 'Application of the multiplication principle in combinatorics'."
    prompt = f"{m}\nInstruction: {t} /think"
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    test_prompts.append(text)

# --- FIX IS HERE ---
# Correctly extract the input_ids tensor (1D) for each prompt
query_tensor_batch = []
for txt in test_prompts:
    # Tokenize the single text, get the 'input_ids' tensor, and then take the first (and only) item
    # to get a 1D tensor. Move it to device.
    input_ids_tensor = tokenizer(txt, return_tensors="pt")['input_ids'][0].to(device)
    query_tensor_batch.append(input_ids_tensor)

print('query_tensor_batch', query_tensor_batch)

# Generate responses for the entire batch
generated_ids_batch = ppo_trainer.generate(query_tensor_batch, return_prompt=False, **generation_kwargs)
print('generated_ids_batch', generated_ids_batch)

# Process responses and create rewards for the entire batch
rewards_batch = []
for i in range(config.batch_size):
    original_prompt_length = len(query_tensor_batch[i])
    output_ids = generated_ids_batch[i][original_prompt_length:].tolist()

    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    print(f"Test Prompt {i+1} Thinking: {thinking_content}")
    print(f"Test Prompt {i+1} Content: {content}")

    rewards_batch.append(torch.tensor(0.5, dtype=torch.float32, device=model.pretrained_model.device))

train_stats = ppo_trainer.step(query_tensor_batch, generated_ids_batch, rewards_batch)

print("----------------------")


# 5. 批量训练
rewards_list = []
for epoch, batch_texts in enumerate(ppo_trainer.dataloader):
    # This part should be similar to the fix above
    query_tensors = []
    for txt in batch_texts:
        input_ids_tensor = tokenizer(txt, return_tensors="pt")['input_ids'][0].to(model.pretrained_model.device)
        query_tensors.append(input_ids_tensor)

    response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)
    response_texts = [tokenizer.decode(resp, skip_special_tokens=True) for resp in response_tensors]

    rewards = []
    for idx, r in enumerate(response_texts):
        # FIX: Explicitly cast to float32
        rewards.append(torch.tensor(1.0 + idx, dtype=torch.float32, device=model.pretrained_model.device))


    train_stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

    print(f"Epoch {epoch}, Generated Responses: {response_texts}")
    print(f"Epoch {epoch}, Reward Sum: {sum(rewards).item()}")
    rewards_list.append(sum(rewards).item())
