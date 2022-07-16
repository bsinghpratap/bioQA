from tqdm import tqdm
import torch
import torch.nn as nn
import transformers
import wandb
from copy import deepcopy
import os
from collections import Counter
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


def evaluate(model, data_loader, objective_f, device):
    model.eval()
    dict_result = {'actual': [],
                   'preds': []}

    # print('\nStarting model evaluation:')
    eval_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # unroll features
            dict_result['actual'] += batch['encoder_labels'].numpy().tolist()
            input_batch = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask']
            }
            input_batch = {k: v.to(device) for k, v in input_batch.items()}

            # forward pass
            logits = model(input_batch)

            # calculate loss
            # print(logits.shape)
            # print(batch['encoder_labels'].shape)
            eval_loss += objective_f(logits, batch['encoder_labels'].to(device)).item()

            # update
            dict_result['preds'] += np.argmax(logits.detach().cpu().numpy(), axis=1).tolist()

    # update
    dict_result['actual'] = [x for x in dict_result['actual']]
    dict_result['loss'] = eval_loss / (batch_idx + 1)

    return dict_result


def get_performance(
        actual_,
        preds_,
        dict_mapping
):
    results = {}

    # accuracy, precision, recall, f1
    results['metrics'] = classification_report(
        actual_,
        preds_,
        output_dict=True,
        zero_division=0,
    )
    for name_, cls_ in dict_mapping.items():
        if not str(cls_) in results['metrics']:
            results['metrics'][str(cls_)] = {'precision': 0}
            #print(f"\nUnique gold labels in the current batch are: {list(set(actual_))}")
            #print(f"Unique predicted labels are: {list(set(preds_))}")

    # confusion matrix
    results['confusion_matrix'] = pd.DataFrame(
        confusion_matrix(
            actual_,
            preds_
        )
    )

    # counter
    results['actual_counter'] = Counter(actual_)
    results['prediction_counter'] = Counter(preds_)

    return results

def test_(
        model,
        test_loader,
        loss_fct,
        label2id,
        epoch,
        global_step,
        device,
):
    test_predictions = evaluate(
        model=model,
        data_loader=test_loader,
        objective_f=loss_fct,
        device=device,
    )
    best_test_results = get_performance(
        actual_=test_predictions['actual'],
        preds_=test_predictions['preds'],
        dict_mapping=label2id
    )

    #
    wandb.log(
        {
            "test/precision": best_test_results['metrics']['macro avg']['precision'],
            "test/recall": best_test_results['metrics']['macro avg']['recall'],
            "test/f1": best_test_results['metrics']['macro avg']['f1-score'],
            "test/accuracy": best_test_results['metrics']['accuracy'],
            "epoch": epoch,

            "test/precision_yes": best_test_results['metrics']['0']['precision'],
            "test/precision_no": best_test_results['metrics']['1']['precision'],
            "test/precision_maybe": best_test_results['metrics']['2']['precision'],
        },
        step=global_step,
    )

    return best_test_results

def train_(
        training_phase,
        fold_idx,
        train_loader,
        val_loader,
        loss_fct,
        model,
        model_name,
        optimizer,
        scheduler,
        num_epochs,
        gradient_accumulation_steps,
        eval_every_steps,
        path_models,
        device,
        label2id,
        debug,
):
    # train
    best_model = None
    best_test_results = None
    best_f1_eval = 0
    best_val_results = None
    global_step = 0
    loss_log = 0

    for each_epoch in tqdm(range(num_epochs)):
        model.train()
        for batch_idx, batch in enumerate(train_loader):

            # unroll inputs and sent to device
            input_batch = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask']
            }
            input_batch = {k: v.to(device) for k, v in input_batch.items()}

            # forward pass
            logits = model(input_batch)

            # calculate loss
            loss = loss_fct(logits, batch['encoder_labels'].to(device))
            loss_log += loss

            # backpropagation
            loss.backward()

            # update parameters and lr
            if ((batch_idx + 1) % gradient_accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
                global_step += 1

                # par update and clean grads
                optimizer.step()
                optimizer.zero_grad()
                model.zero_grad()

                # log info to wandb
                wandb.log(
                    {
                        "train/loss": loss_log / gradient_accumulation_steps,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "epoch": each_epoch,
                },
                step = global_step,
                )

                # update logged value
                loss_log = 0

                # update LR
                if ((batch_idx + 1) % gradient_accumulation_steps == 0):
                    scheduler.step()

            # evaluation
            if global_step % eval_every_steps == 0:
                # evaluate model
                val_predictions = evaluate(
                    model=model,
                    data_loader=val_loader,
                    objective_f=loss_fct,
                    device=device,
                )
                val_results = get_performance(
                    actual_=val_predictions['actual'],
                    preds_=val_predictions['preds'],
                    dict_mapping=label2id
                )

                # log info to wandb
                wandb.log(
                    {
                        "eval/precision": val_results['metrics']['macro avg']['precision'],
                        "eval/recall": val_results['metrics']['macro avg']['recall'],
                        "eval/f1": val_results['metrics']['macro avg']['f1-score'],
                        "eval/accuracy": val_results['metrics']['accuracy'],
                        "eval/loss": val_predictions['loss'],
                        "epoch": each_epoch,

                        "eval/precision_yes": val_results['metrics']['0']['precision'],
                        "eval/precision_no": val_results['metrics']['1']['precision'],
                        "eval/precision_maybe": val_results['metrics']['2']['precision'],
                    },
                    step=global_step,
                )

                # update best model
                if best_f1_eval < val_results['metrics']['weighted avg']['f1-score']:
                    # best_model = deepcopy(model).to(device)
                    best_val_results = deepcopy(val_results)
                    best_f1_eval = val_results['metrics']['weighted avg']['f1-score']

                    # save model
                    torch.save(
                        model.state_dict(),
                        os.path.join(path_models, f"{model_name.replace('/', '-')}_{training_phase}_fold{fold_idx}.pt")
                    )

        if debug:
            if each_epoch > 3:
                break

    # load the best model
    model.load_state_dict(
        torch.load(
            os.path.join(path_models, f"{model_name.replace('/', '-')}_{training_phase}_fold{fold_idx}.pt")
        )
    )

    return model, each_epoch, global_step


# function for collecting all predictions on the input dataset
def label_data(
        model,
        loader,
        device,
        debug,

):
    model.eval()

    #
    dict_results = {}
    all_preds = []
    for batch_idx, batch_ in tqdm(enumerate(loader)):
        with torch.inference_mode():

            # unroll features
            input_batch = {
                'input_ids': batch_['input_ids'],
                'attention_mask': batch_['attention_mask']
            }
            input_batch = {k: v.to(device) for k, v in input_batch.items()}

            # forward pass
            logits = model(input_batch)

            # update
            preds = np.argmax(logits.detach().cpu().numpy(), axis=1).tolist()
            all_preds += preds
            ids_ = batch_['ids'].numpy().tolist()
            for id_idx, id_ in enumerate(ids_):
                dict_results[str(id_)] = {'custom_label': preds[id_idx]}

            #
            if debug:
                if batch_idx > 3:
                    break

    # get distribution of predicted labels
    count = {}
    count['yes'] = (np.array(all_preds) == 0).sum()
    count['no'] = (np.array(all_preds) == 1).sum()
    count['maybe'] = (np.array(all_preds) == 2).sum()
    dist_class = {}
    for i in ['yes', 'no', 'maybe']:
        dist_class[i] = count[i] / len(all_preds)

    return dict_results, dist_class


