## data_augmentation.py

import json
import pandas as pd
from BERT_augmentation import BERT_Augmentation
from adverb_augmentation import AdverbAugmentation
from data_loader import load_megumin_dialogues_to_blessing, load_megumin_dialogues_to_flame, load_megumin_qa_dialogues, extract_github_dialogues
from tqdm import tqdm

## augmentation 객체 생성
BERT_aug = BERT_Augmentation()
random_masking_insertion = BERT_aug.random_masking_insertion

adverb_aug = AdverbAugmentation()
adverb_gloss_replacement = adverb_aug.adverb_gloss_replacement

def augment_data(dialogue_list, ratio=0.1):
    augmented_data = []

    for item in tqdm(dialogue_list, desc="Augmenting Data"):
        scene_context = item['scene_context']
        context = item['context']
        response = item['response']
        
        ## random masking replacement와 insertion을 각각 적용
        augmented_context_insertion = random_masking_insertion(context, ratio)
        augmented_context_adverb = adverb_gloss_replacement(context)
        
        augmented_response_insertion = random_masking_insertion(response, ratio)
        augmented_response_adverb = adverb_gloss_replacement(response)
        
        ## 원본 데이터 추가
        augmented_data.append({
            'scene_context': scene_context,            
            'context': context,
            'response': response
        })

        ## 첫번째 증강된 데이터 추가
        augmented_data.append({
            'scene_context': scene_context,
            'context': augmented_context_insertion,
            'response': augmented_response_insertion
        })

        ## 두번째 증강된 데이터 추가
        augmented_data.append({
            'scene_context': scene_context,
            'context': augmented_context_adverb,
            'response': augmented_response_adverb
        })

    return augmented_data

## 생성된 데이터 저장 함수
def save_dialogues_as_json(dialogues, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dialogues, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    blessing_directory_path = "Json_Data/blessing_folder"
    flame_directory_path = "Json_Data/flame_dialogue.json"
    after_directory_path = "Json_Data/after_dialogue.json"
    movie_directory_path = "Json_Data/movie_dialogue.json"
    github_directory_path = "Json_Data/github_dialogue.json"
    qa_directory_path = "Json_Data/dialogues.json"
    # huggingface_directory_path = "Json_Data/huggingface_dataset.json"

    blessing_dialogues = load_megumin_dialogues_to_blessing(blessing_directory_path)
    flame_dialogues = load_megumin_dialogues_to_flame(flame_directory_path)
    after_dialogues = load_megumin_dialogues_to_flame(after_directory_path)
    movie_dialogues = load_megumin_dialogues_to_flame(movie_directory_path)
    github_dialogue = extract_github_dialogues(github_directory_path)
    qa_dialogues = load_megumin_qa_dialogues(qa_directory_path)
    # huggingface_dialogues = load_megumin_qa_dialogues(huggingface_directory_path)

    blessing_dialogues.extend(flame_dialogues)
    blessing_dialogues.extend(after_dialogues)
    blessing_dialogues.extend(movie_dialogues)
    blessing_dialogues.extend(github_dialogue)
    blessing_dialogues.extend(qa_dialogues)
    # blessing_dialogues.extend(huggingface_dialogues)

    augmentation_dataset = augment_data(blessing_dialogues, ratio=0.2)
    save_dialogues_as_json(augmentation_dataset, "aug_output.json")