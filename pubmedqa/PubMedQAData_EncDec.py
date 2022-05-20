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
from functools import partial


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
        self.source_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)#, TOKENIZERS_PARALLELISM=False)
        self.target_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)#, TOKENIZERS_PARALLELISM=False)

        # STEP 4: tokenize dataset
        self.dataset_train = QADataset(
            list_data=data['train'] if not debug else data['train'][0:debug_size],
            source_tokenizer=self.source_tokenizer,
            target_tokenizer=self.target_tokenizer,
            label2id=self.label2id,
            max_length=max_sequence_length
        )
        self.dataset_test = QADataset(
            list_data=data['test'] if not debug else data['test'][0:debug_size],
            source_tokenizer=self.source_tokenizer,
            target_tokenizer=self.target_tokenizer,
            label2id=self.label2id,
            max_length=max_sequence_length
        )
        self.dataset_validation = QADataset(
            list_data=data['validation'] if not debug else data['validation'][0:debug_size],
            source_tokenizer=self.source_tokenizer,
            target_tokenizer=self.target_tokenizer,
            label2id=self.label2id,
            max_length=max_sequence_length
        )
        
        # define collation function
        collation_wrapper = partial(
            self.collation_f,
            source_pad_token_id=self.source_tokenizer.pad_token_id,
            target_pad_token_id=self.target_tokenizer.pad_token_id,
        )
        

        # STEP 5: get dataloaders for training and testing
        self.dataloader_train = DataLoader(
            self.dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collation_wrapper,
        )
        self.dataloader_test = DataLoader(
            self.dataset_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collation_wrapper,
        )
        self.dataloader_validation = DataLoader(
            self.dataset_validation,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collation_wrapper,
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
            dict_data_: dict
    ):

        list_data = []
        for idx in range(len(dict_data_['question'])):

            # try extracting final decision
            try:
                final_decision = dict_data_['final_decision'][idx]
            except:
                final_decision = 'no label'
            
            if not final_decision in ['no label', 'maybe']:
                # create data instance
                instance = {
                    'source_question': dict_data_['question'][idx],
                    'source_context': ''.join(dict_data_['context'][idx]['contexts']),
                    'target_answer': dict_data_['long_answer'][idx],
                    'gold_label': final_decision,
                }
                list_data.append(instance)

        return list_data
    
    def collation_f(
        self,
        batch,
        source_pad_token_id, 
        target_pad_token_id,
    ):
        
        input_ids_list = [ex["input_ids"] for ex in batch]
        decoder_input_ids_list = [ex["decoder_input_ids"] for ex in batch]
        decoder_labels_list = [ex["decoder_labels"] for ex in batch]
        encoder_label_list = [ex['gold_label'] for ex in batch]

        collated_batch = {
            "input_ids": self.pad(input_ids_list, source_pad_token_id),
            "encoder_labels": torch.LongTensor(encoder_label_list).flatten(end_dim=1),
            "decoder_input_ids": self.pad(decoder_input_ids_list, target_pad_token_id),
            "decoder_labels": self.pad(decoder_labels_list, target_pad_token_id),

        }
        collated_batch["attention_mask"] = collated_batch["input_ids"] != source_pad_token_id

        return collated_batch 
    
    def pad(
        self,
        sequence_list, 
        pad_id
    ):
        """Pads sequence_list to the longest sequence in the batch with pad_id.

        Args:
            sequence_list: a list of size batch_size of numpy arrays of different length
            pad_id: int, a pad token id

        Returns:
            torch.LongTensor of shape [batch_size, max_sequence_len]
        """
        max_len = 512#max(len(x) for x in sequence_list)
        padded_sequence_list = []
        for sequence in sequence_list:
            padding = [pad_id] * (max_len - len(sequence))
            padded_sequence = sequence + padding
            padded_sequence_list.append(padded_sequence)

        return torch.LongTensor(padded_sequence_list)


class QADataset(Dataset):
    """Class to load the dataset and get batches of paras"""

    def __init__(
            self,
            list_data: list,
            source_tokenizer,
            target_tokenizer,
            max_length: int,
            label2id: dict,
    ):

        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.data = list_data
        #self.pad_token = self.tokenizer.vocab[self.tokenizer._pad_token]
        #self.pad_token_id = self.tokenizer.pad_token_id

    def __len__(self):
        """Return length of dataset."""
        return self.data.__len__()

    def __getitem__(self, i):
        """Return sample from dataset at index i."""
        
        example = self.data[i]
        inputs = self.source_tokenizer.encode_plus(
            example['source_question'],
            example['source_context'],
            add_special_tokens=True,
            truncation='only_second',
            max_length=self.max_length
        )
        targets = self.target_tokenizer.encode_plus(
            example['target_answer'],
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length
        )
        targets['decoder_labels'] = targets['input_ids'][1:] + [self.target_tokenizer.pad_token_id]

        return_dict = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'decoder_input_ids': targets['input_ids'],
            'decoder_labels': targets['decoder_labels'],
            'decoder_attention_mask': targets['attention_mask'],
            'gold_label': [self.label2id[example['gold_label']]],
        }
        

        return return_dict