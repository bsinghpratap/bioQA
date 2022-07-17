import json

import utility_fns as utils_f
from PubMedQAData import QADataLoader
import trainer
import torch
import numpy as np
import random
import argparse
import wandb
import os

def parse_args():
    """
    This function creates argument parser and parses the scrip input arguments.

    """
    parser = argparse.ArgumentParser(description="PubMedQA self-learning")

    # Paths
    parser.add_argument(
        "--path_data",
        type=str,
        default=r'/Users/vijetadeshpande/Downloads/BioNLP Lab/Datasets/QA/pubmedqa/data',
        #required=True,
    )
    parser.add_argument(
        "--path_models",
        type=str,
        default='results',
        #required=True,
    )

    # data
    parser.add_argument(
        "--folds_total",
        type=int,
        default=10,
        # required=True,
    )
    parser.add_argument(
        "--folds_sample_size",
        type=int,
        default=3,
        # required=True,
    )
    parser.add_argument(
        "--device",
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    parser.add_argument(
        "--cuda_device_index",
        type=str,
        default=str(3),
    )


    # language model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default='prajjwal1/bert-tiny',
        #required=True,
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default='prajjwal1/bert-tiny',
        #required=True,
    )

    # training arguments
    parser.add_argument(
        "--fixed_seed_value",
        type=int,
        default=3,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="batch size for phase-1/3",
    )
    parser.add_argument(
        "--maximum_sequence_length",
        type=int,
        default=512,
        help="maximum input sequence length",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="number of epochs for phase-1/3",
    )
    parser.add_argument(
        "--eval_every_steps",
        type=int,
        default=1,
        help="evaluation frequency",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="gradient accumulation for phase-1/3",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="maximum value of learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=10,
        help="weight decay for Adam",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-8,
        help="epsilon parameter for Adam",
    )
    parser.add_argument(
        "--scheduler_warmup",
        type=float,
        default=0.2,
        help="percentage of total steps for learning rate warmup",
    )
    parser.add_argument(
        "--no_decay_layers",
        type=list,
        default=['bias', 'LayerNorm.weight'],
        help="layers in the encoder that will have weight decay value of zero",
    )

    # visualization arguments
    parser.add_argument(
        "--wandb_project",
        default="PubMedQA_tiny",
        help="wandb project name to log metrics to"
    )

    # additional arguments
    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="when true, only 8 examples are selected to train model"
    )

    args = parser.parse_args()

    return args


def init_training(
        training_phase,
        fold_idx,
        args
):
    #
    args.training_phase = training_phase
    args.fold_idx = fold_idx

    #
    print("\n")
    print("=" * 30)
    print(f"\nStarting training {training_phase} model")
    print("=" * 30)
    print("\n")

    #
    run = wandb.init(
        project=args.wandb_project,
        config=args,
        name=f"{training_phase}_fold{fold_idx}",
    )

    # dataloader
    dataloaders = QADataLoader(
        training_phase=training_phase,
        fold_idx=args.fold_idx,
        path_data=args.path_data,
        fixed_seed_value=args.fixed_seed_value,
        label2id=None,
        tokenizer_name=args.tokenizer_name,
        batch_size=args.batch_size if training_phase in ["phase-1", "phase-3"] else 32,
        debug=args.debug,
    )

    # model
    model = utils_f.get_model(
        model_name=args.model_name,
        fold_idx=args.fold_idx,
        num_classes=args.num_classes,
        class_dist=args.class_dist,
        training_phase=training_phase,
        path_models=args.path_models,
        device=args.device,
    )

    # objective function
    loss_fct = utils_f.get_loss_function(
        class_weights=args.class_weights,
        id2label=dataloaders.id2label,
        device=args.device,
    )

    # solution method
    optimizer = utils_f.get_optimizer(
        model=model,
        no_decay_layers=args.no_decay_layers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_epsilon=args.adam_epsilon,
    )

    # learning rate scheduler
    scheduler = utils_f.get_lr_scheduler(
        optimizer=optimizer,
        train_loader=dataloaders.dataloader_train,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        scheduler_warmup=args.scheduler_warmup,
    )

    # train
    best_model, last_epoch, last_training_step = trainer.train_(
        training_phase=training_phase,
        fold_idx=args.fold_idx,
        train_loader=dataloaders.dataloader_train,
        val_loader=dataloaders.dataloader_validation,
        loss_fct=loss_fct,
        model=model,
        model_name=args.model_name,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs if training_phase in ["phase-1", "phase-3"] else 2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_every_steps=args.eval_every_steps,
        path_models=args.path_models,
        device=args.device,
        label2id=dataloaders.label2id,
        debug=args.debug,
    )

    #
    print("\n")
    print("=" * 30)
    print(f"\nStarting testing of {training_phase} model")
    print("=" * 30)
    print("\n")

    # test
    best_test_result = trainer.test_(
        model=best_model,
        test_loader=dataloaders.dataloader_test,
        loss_fct=loss_fct,
        label2id=dataloaders.label2id,
        epoch=last_epoch,
        global_step=last_training_step,
        device=args.device,
    )
    utils_f.save_test_performance_results(
        results=best_test_result,
        model_name=args.model_name,
        training_phase=training_phase,
        fold_idx=fold_idx,
        path_models=args.path_models,
    )

    #
    if training_phase == "phase-2":
        print("\n")
        print("=" * 30)
        print(f"\nTesting phase-2 model on manually labeled test set")
        print("=" * 30)
        print("\n")

        #
        dataloaders = QADataLoader(
            training_phase="phase-1",
            fold_idx=args.fold_idx,
            path_data=args.path_data,
            fixed_seed_value=args.fixed_seed_value,
            label2id=None,
            tokenizer_name=args.tokenizer_name,
            batch_size=args.batch_size,
            debug=args.debug,
        )
        # test
        best_test_result = trainer.test_(
            model=best_model,
            test_loader=dataloaders.dataloader_test,
            loss_fct=loss_fct,
            label2id=dataloaders.label2id,
            epoch=last_epoch+1,
            global_step=last_training_step+1,
            device=args.device,
        )
        utils_f.save_test_performance_results(
            results=best_test_result,
            model_name=args.model_name,
            training_phase="phase-2_on_labeled",
            fold_idx=fold_idx,
            path_models=args.path_models,
        )

    # end the wandb run
    run.finish()

    #
    if training_phase == "phase-1":
        print("\n")
        print("=" * 30)
        print(f"\nStarting data annotation with model developed in phase-1")
        print("=" * 30)
        print("\n")

        dataloaders = QADataLoader(
            training_phase="annotation",
            fold_idx=args.fold_idx,
            path_data=args.path_data,
            fixed_seed_value=args.fixed_seed_value,
            label2id=None,
            tokenizer_name=args.tokenizer_name,
            batch_size=512,
            debug=False,
        )
        model = utils_f.get_model(
            model_name=args.model_name,
            fold_idx=args.fold_idx,
            num_classes=args.num_classes,
            class_dist=args.class_dist,
            training_phase="annotation",
            path_models=args.path_models,
            device=args.device,
        )
        predictions, dist_class = trainer.label_data(
            model=model,
            loader=dataloaders.dataloader_train,
            device=args.device,
            debug=args.debug,
        )
        with open(os.path.join(args.path_models, f'ditribution_of_annotated_data_{fold_idx}.json'), 'w') as f:
            json.dump(dist_class, f, indent=4)

        #
        print("\n")
        print("="*30)
        print(f"\nDistribution of artificially labeled data is: \n{dist_class}")
        print(f"\nSaving model annotated data")
        print("=" * 30)
        print('\n')

        #
        utils_f.save_model_labeled_data(
            path_data=args.path_data,
            fold_idx=args.fold_idx,
            id2label=dataloaders.id2label,
            predictions=predictions,
        )


    return


def main():

    # args for hyper parameters
    args = parse_args()

    # make sure output directory exists
    if not os.path.exists(args.path_models):
        os.makedirs(args.path_models)

    # fix seed
    utils_f.fix_all_seed(args.fixed_seed_value)

    # set device to use
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device_index

    # select the folds/seed to use
    selected_folds = utils_f.sample_folds(
        path_data=args.path_data,
        seed=args.fixed_seed_value,
        folds_total=args.folds_total,
        folds_sample_size=args.folds_sample_size
    )

    # calculate class distribution and class weight based on the training data we have
    num_classes, class_dist, class_weights = utils_f.get_class_properties(args.path_data)
    args.num_classes = num_classes
    args.class_dist = class_dist
    args.class_weights = class_weights

    #
    for fold in selected_folds:
        print("\n")
        print("=" * 30)
        print(f"\nStarting sequential training with the fold{fold} version of labeled data")
        print("=" * 30)
        print("\n")
        args.fold_idx = fold

        # phase-1
        init_training("phase-1", fold, args)

        # phase-2
        init_training("phase-2", fold, args)

        # phase-3
        init_training("phase-3", fold, args)

    return


if __name__ == '__main__':
    main()