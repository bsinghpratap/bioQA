import pandas as pd
import torch
import numpy as np
import json
from tqdm import tqdm
import os
from copy import deepcopy
from PubMedQAModel import QAModel
from torch.nn import CrossEntropyLoss
from transformers import (BertPreTrainedModel, BertModel, AdamW, get_linear_schedule_with_warmup,
                          RobertaPreTrainedModel, RobertaModel,
                          AutoTokenizer, AutoModel, AutoConfig)
from collections import Counter
import random

def sample_fold_indices(
        seed: int,
        total_fold_choices: int = 10,
        sample_size: int = 3,
):

    #
    np.random.seed(seed=seed)
    final_folds = np.random.choice(range(total_fold_choices), size=sample_size).astype(int)

    return final_folds

def sample_folds(
        path_data: str,
        seed: int,
        folds_total: int = 10,
        folds_sample_size: int = 3,
):

    # sample fold indices
    fold_indices = sample_fold_indices(
        seed=seed,
        total_fold_choices=folds_total,
        sample_size=folds_sample_size
    )

    # read data
    folds = {}
    for idx in fold_indices:
        path_read = os.path.join(path_data, 'pqal_fold' + str(idx))

        # read data
        with open(os.path.join(path_read, 'train_set.json'), 'r') as f:
            fold_train = json.load(f)
        with open(os.path.join(path_read, 'dev_set.json'), 'r') as f:
            fold_validation = json.load(f)

        folds[idx] = {
            'train': deepcopy(fold_train),
            'validation': deepcopy(fold_validation)
        }

    return folds

def annotate_with_model(
        model,
        batch,
        device,
):
    model.eval()

    #
    with torch.inference_mode():
        # unroll features
        input_batch = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask']
        }
        input_batch = {k: v.to(device) for k, v in input_batch.items()}

        # forward pass
        logits = model(input_batch)

        # get predicted labels
        labels = torch.argmax(logits, dim=-1, keepdim=False)

    return labels

def get_model(
        model_name: str,
        fold_idx: int,
        fixed_seed_value: int,
        num_classes: int,
        class_dist: dict,
        training_phase: str,
        path_models: str,
        device
):

    #
    model = QAModel(
        model_name=model_name,
        num_classes=num_classes,
    )

    #
    for name, param in model.named_parameters():
        if 'classifier.weight' in name:
            torch.nn.init.zeros_(param.data)
        elif 'classifier.bias' in name:
            param.data = torch.tensor([class_dist['yes'], class_dist['no'], class_dist['maybe']]).float()

    if training_phase == "phase-3":
        model.load_state_dict(torch.load(os.path.join(path_models, f"{model_name.replace('/', '-')}_phase-2_fold{fold_idx}_seed{fixed_seed_value}.pt")))
    elif training_phase == "annotation":
        model.load_state_dict(torch.load(os.path.join(path_models, f"{model_name.replace('/', '-')}_phase-1_fold{fold_idx}_seed{fixed_seed_value}.pt")))

    model = model.to(device)

    return model

def get_class_properties(
        path_data: str,
):

    with open(os.path.join(path_data, 'pqal_fold0', 'train_set.json'), 'r') as f:
        train_set = json.load(f)

    with open(os.path.join(path_data, 'pqal_fold0', 'dev_set.json'), 'r') as f:
        dev_set = json.load(f)

    #
    data = {**train_set, **dev_set}
    class_count = Counter([data[x]['final_decision'] for x in data])
    num_classes = len(class_count)

    #
    denominator = sum([class_count[x] for x in class_count])
    class_weight = {}
    class_dist = {}
    for cls in class_count:
        class_weight[cls] = np.log(num_classes * denominator / class_count[cls])
        class_dist[cls] = class_count[cls] / denominator

    return num_classes, class_dist, class_weight

def get_loss_function(
        class_weights: dict,
        id2label: dict,
        device,
):

    #
    sorted_class_ids = sorted(list(id2label.keys()))
    sorted_class_weights = [class_weights[id2label[i]] for i in sorted_class_ids]

    #
    loss_fct = CrossEntropyLoss(
        weight=torch.tensor(sorted_class_weights).float().to(device),
        ignore_index=-100,
    )

    return loss_fct


def get_grouped_parameters(
        model,
        no_decay_layers,
        weight_decay,
):
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay_layers)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay_layers)],
         'weight_decay': 0.0}
    ]

    return optimizer_grouped_parameters


def get_optimizer(
        model,
        no_decay_layers,
        learning_rate,
        weight_decay,
        adam_epsilon,
):
    optimizer = torch.optim.AdamW(
        get_grouped_parameters(model, no_decay_layers, weight_decay),
        lr=learning_rate,
        eps=adam_epsilon,
    )

    return optimizer

def get_lr_scheduler(
        optimizer,
        train_loader,
        num_epochs,
        gradient_accumulation_steps,
        scheduler_warmup,
):
    total_steps = int(len(train_loader) / gradient_accumulation_steps) * num_epochs
    warmup_steps = int(scheduler_warmup * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    return scheduler

def fix_all_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    return

def save_model_labeled_data(
        path_data: str,
        fold_idx: int,
        id2label: dict,
        predictions,
):

    with open(os.path.join(path_data, 'ori_pqaa.json'), 'r') as f:
        a_ = json.load(f)
    with open(os.path.join(path_data, 'ori_pqau.json'), 'r') as f:
        u_ = json.load(f)

    model_labeled_data = {}
    for data_ in [a_, u_]:
        for id_idx, id_ in enumerate(data_):
            model_labeled_data[id_] = data_[id_]
            if id_ in predictions:
                model_labeled_data[id_]['custom_label'] = id2label[predictions[id_]['custom_label']]
            else:
                model_labeled_data[id_]['custom_label'] = ''

    #
    with open(os.path.join(path_data, f'model_annotated_data_fold{fold_idx}.json'), 'w') as f:
        json.dump(model_labeled_data, f, indent=4)

    return

def save_test_performance_results(
        results: dict,
        model_name: str,
        training_phase: str,
        fold_idx: int,
        path_models: str,
):
    results_to_save = {}
    for k_ in results['metrics']:
        if k_ in ['macro avg', 'accuracy']:
            results_to_save[k_] = results['metrics'][k_]

    with open(os.path.join(path_models, f'test_results_{model_name.replace("/", "-")}_{training_phase}_fold{fold_idx}.json'), 'w') as f:
        json.dump(results_to_save, f, indent=4)

    results['confusion_matrix'].to_csv(
        os.path.join(path_models, f'confusion_matrix_{model_name.replace("/", "-")}_{training_phase}_fold{fold_idx}.csv')
    )

    return
