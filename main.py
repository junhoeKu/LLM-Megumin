## main.py
import json
import wandb
import torch
from training import tokenize_dialogues, fine_tune_model
from data_loader import load_megumin_dialogues_to_blessing, load_megumin_dialogues_to_flame, \
    load_megumin_qa_dialogues, extract_github_dialogues


if __name__ == "__main__":
    wandb.login(key="15f21c20fccb0099e4a274798c7c257ccc39988a")
    with open('Json_Data/augmented_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    torch.cuda.empty_cache()

    ## 2. 메구밍 대사 토크나이징
    tokenized_megumin_dataset = tokenize_dialogues(data)
    torch.cuda.empty_cache()

    ## 3. Fine Tuning 진행
    fine_tune_model(tokenized_megumin_dataset)