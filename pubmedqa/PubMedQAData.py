import torch
import random
from torch.utils.data import Dataset
import datasets
from torch.utils.data import DataLoader
import scipy.sparse
from pdb import set_trace as breakpoint
from collections import Counter
from random import shuffle
import json
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split


class QADataLoader():

    def __init__(
            self,
            datasets_name: str = None,
            datasets_config: str = None,
            filepath_data: str = '',
            label2id: dict = None,
            filepath_label2id: str = 'label2id.json',
            tokenizer_name: str = 'bert-base-uncased',
            max_sequence_length: int = 512,
            batch_size: int = 16,
            debug: bool = False,
            debug_size: int = 8,
    ):
        """This class saves dataloader of each split as attributes"""

        # STEP 1: read data and convert to list_of_dict format
        if datasets_name is None and datasets_config is None:
            with open(filepath_data, 'r') as f:
                data = json.load(f)
        else:
            data = datasets.load_dataset(datasets_name, datasets_config)
        data = self.get_splits(data)
        for split in data:
            data[split] = self.get_list_data(data[split])

        # STEP 2: in addition to data, we need label maps i.e. a dict to convert
        # label text to an integer and an integer back to it's class text
        if label2id is None:
            with open(filepath_label2id, 'r') as f:
                self.label2id = json.load(f)
        else:
            self.label2id = label2id

        # the reverse map i.e. id2label is useful in evaluation/interact
        # when we are required to convert prediction (an integer) to class name
        self.id2label = self.get_id2label(self.label2id)

        # save number of classes for both labels
        self.num_classes = self.get_num_classes(self.label2id)

        # STEP 3: define tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # STEP 4: tokenize dataset
        self.dataset_train = QADataset(
            list_data=data['train'] if not debug else data['train'][0:debug_size],
            tokenizer=self.tokenizer,
            label2id=self.label2id,
            max_length=max_sequence_length
        )
        self.dataset_test = QADataset(
            list_data=data['test'] if not debug else data['test'][0:debug_size],
            tokenizer=self.tokenizer,
            label2id=self.label2id,
            max_length=max_sequence_length
        )
        self.dataset_validation = QADataset(
            list_data=data['validation'] if not debug else data['validation'][0:debug_size],
            tokenizer=self.tokenizer,
            label2id=self.label2id,
            max_length=max_sequence_length
        )

        # STEP 5: get dataloaders for training and testing
        self.dataloader_train = DataLoader(
            self.dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        self.dataloader_test = DataLoader(
            self.dataset_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        self.dataloader_validation = DataLoader(
            self.dataset_validation,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        return

    def get_id2label(
            self,
            label2id: dict
    ) -> dict:
        id2label = {}
        for key_, val_ in label2id.items():
            id2label[val_] = key_

        return id2label

    def get_num_classes(
            self,
            id2label: dict
    ) -> int:

        return len(id2label.keys())

    def get_splits(
            self,
            data_in
    ):
        data_out = {}
        data_train, data_test = train_test_split(data_in['train'])
        data_out['train'] = data_train
        data_out['test'] = data_test
        data_out['validation'] = data_test

        return data_out

    def get_list_data(
            self,
            dict_data: dict
    ):

        list_data = []
        for idx in range(len(dict_data['question'])):
            instance = {
                'sentence1': dict_data['question'][idx],
                'sentence2': ''.join(dict_data['context'][idx]['contexts']),
                'gold_label': dict_data['final_decision'][idx]
            }
            list_data.append(instance)

        return list_data


class QADataset(Dataset):
    """Class to load the dataset and get batches of paras"""

    def __init__(
            self,
            list_data: list,
            tokenizer,
            max_length: int,
            label2id: dict,
    ):

        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.data = list_data
        self.pad_token = self.tokenizer.vocab[self.tokenizer._pad_token]

    def __len__(self):
        """Return length of dataset."""
        return self.data.__len__()

    def __getitem__(self, i):
        """Return sample from dataset at index i."""
        example = self.data[i]
        inputs = self.tokenizer.encode_plus(
            example['sentence1'],
            example['sentence2'],
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length
        )

        #
        input_ids = inputs["input_ids"]
        input_ids = input_ids[:self.max_length]
        attention_mask = [1] * len(input_ids)
        padding_length = self.max_length - len(input_ids)
        input_ids = input_ids + ([self.pad_token] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)

        #
        if 'token_type_ids' in inputs:
            token_type_ids = inputs["token_type_ids"]
            token_type_ids = token_type_ids[:self.max_length]
            token_type_ids = token_type_ids + ([self.tokenizer._pad_token] * padding_length)

        assert len(input_ids) == self.max_length, "Error with input length {} vs {}".format(len(input_ids),
                                                                                            self.max_length)
        assert len(attention_mask) == self.max_length, "Error with input length {} vs {}".format(len(attention_mask),
                                                                                                 self.max_length)

        label = self.label2id[example['gold_label']]
        return_dict = {
            'input_ids': torch.LongTensor(input_ids),
            'attention_mask': torch.LongTensor(attention_mask),
            'label': torch.LongTensor([label])
        }

        if 'token_type_ids' in inputs:
            assert len(token_type_ids) == self.max_length, "Error with input length {} vs {}".format(
                len(token_type_ids), self.max_length)
            return_dict['token_type_ids'] = torch.LongTensor(token_type_ids)

        return return_dict