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
from tqdm import tqdm


class QADataLoader():

    def __init__(
            self,
            datasets_name: str = None,
            datasets_config: str = None,
            filepath_data: str = r'./ori_pqaa.json',
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
            """
            print('Reading unlabeled and artificially labeled subsets of the data')
            with open(filepath_data, 'r') as f:
                a_ = json.load(f)
            with open('ori_pqau.json', 'r') as f:
                u_ = json.load(f)
            data = {'train': {**a_, **u_}}
            print(f'length of artificially labled data: {len(a_)}')
            print(f'length of unlabeled data: {len(u_)}')
            t_set = data['train']
            print(f'length of combined data: {len(t_set)}')
            """
            with open('model_labeled_data.json', 'r') as f:
                data = {'train': json.load(f)}
        else:
            data = {}
            with open('train_set.json', 'r') as f:
                data['train'] = json.load(f)
            with open('dev_set.json', 'r') as f:
                data['validation'] = json.load(f)
            with open('test_set.json', 'r') as f:
                data['test'] = json.load(f)
            
        #
        for split in data:
            #data[split] = self.get_list_data(data[split])
            data[split] = self.get_list_data_file(data[split])
        
        """
        # @TODO: remove following lines once done with creating artificial labels
        data_unl = datasets.load_dataset(datasets_name, 'pqa_unlabeled')
        for split in data_unl:
            data[split] += self.get_list_data(data_unl[split])
        """        
        #
        if ('test' not in data) or ('validation' not in data):
            data = self.get_splits(data)
        data['train'] = self.oversample(data['train'], {'no': 1, 'maybe': 2, 'yes': 1})

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
            shuffle=False,
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
    
    def oversample(self, data_in, class_multi):
        
        data_out = []
        for instance in data_in:
            i_ = 0
            while i_ < class_multi[instance['gold_label']]:
                data_out.append(instance)
                i_ += 1
        #
        random.shuffle(data_out)
        
        return data_out

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
            data_in,
            no_split=False
    ):
        if no_split:
            data_in['test'] = []
            data_in['validation'] = []
            return data_in
        
        #
        data_out = {}
        data_train, data_test = train_test_split(
            data_in['train'],
            test_size=0.01,
            random_state=1,
        )
        data_test, data_val = train_test_split(
            data_test,
            test_size=0.5,
            random_state=1
        )
        data_out['train'] = data_train
        data_out['test'] = data_test
        data_out['validation'] = data_val

        return data_out
    
    def get_list_data_file(
        self,
        dict_data_,
    ):
        #list_ = list(dict_data.values())
        list_data = []
        for idx, id_ in enumerate(dict_data_):
            
            if 'final_decision' in dict_data_[id_]:
                if not dict_data_[id_]['final_decision'] == '':
                    instance = {
                        'source_question': dict_data_[id_]['QUESTION'],
                        'source_context': ' '.join(dict_data_[id_]['CONTEXTS']),
                        'target_answer': dict_data_[id_]['LONG_ANSWER'],
                        'gold_label': dict_data_[id_]['final_decision'], #dict_data_[id_]['custom_label'],
                        'id': id_,
                    }
                    list_data.append(instance)
            """
            else:
                instance = {
                    'source_question': dict_data_[id_]['QUESTION'],
                    'source_context': ' '.join(dict_data_[id_]['CONTEXTS']),
                    'target_answer': dict_data_[id_]['LONG_ANSWER'],
                    'gold_label': 'maybe', # we do not use this label, this is placeholder for unlabeled data
                    'id': id_,
                }
            
                
            
            # try extracting final decision
            try:
                final_decision = dict_data_[id_]['custom_label']
                if not final_decision == '':
                    instance = {
                        'source_question': dict_data_[id_]['QUESTION'],
                        'source_context': ' '.join(dict_data_[id_]['CONTEXTS']),
                        'target_answer': dict_data_[id_]['LONG_ANSWER'],
                        'gold_label': final_decision,
                        'id': id_,
                    }
                    list_data.append(instance)
            except:
                continue
            """
            
        return list_data

    def get_list_data(
            self,
            dict_data_: dict
    ):

        list_data = []
        for idx in tqdm(range(len(dict_data_['question']))):

            # try extracting final decision
            try:
                final_decision = dict_data_['final_decision'][idx]
            except:
                final_decision = 'maybe' # this is just an adjustment, we do not use these 'maybe' labels

            """
            # create data instance
            instance = {
                'source_question': dict_data_['question'][idx],
                'source_context': ' '.join(dict_data_['context'][idx]['contexts']),
                'target_answer': dict_data_['long_answer'][idx],
                'gold_label': final_decision,
            }
            list_data.append(instance)
            
            """
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
        attention_mask_list = [ex["attention_mask"] for ex in batch]
        decoder_input_ids_list = [ex["decoder_input_ids"] for ex in batch]
        decoder_attention_mask_list = [ex["decoder_attention_mask"] for ex in batch]
        decoder_labels_list = [ex["decoder_labels"] for ex in batch]
        encoder_label_list = [ex['gold_label'][0] for ex in batch]
        ids_ = [ex['id'][0] for ex in batch]

        collated_batch = {
            "input_ids": self.pad(input_ids_list, source_pad_token_id),
            "attention_mask": self.pad(attention_mask_list, source_pad_token_id),
            "encoder_labels": torch.LongTensor(encoder_label_list),
            "decoder_input_ids": self.pad(decoder_input_ids_list, target_pad_token_id),
            "decoder_attention_mask": self.pad(decoder_attention_mask_list, target_pad_token_id),
            "decoder_labels": self.pad(decoder_labels_list, target_pad_token_id),
            "ids": torch.LongTensor(ids_),
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
        self.source_pad_token_id = self.source_tokenizer.pad_token_id
        self.target_pad_token_id = self.target_tokenizer.pad_token_id

    def __len__(self):
        """Return length of dataset."""
        return self.data.__len__()

    def __getitem__(self, i):
        """Return sample from dataset at index i."""
        
        example = self.data[i]
        
        # source sequence
        inputs = self.source_tokenizer.encode_plus(
            example['source_question'],
            example['source_context'],
            add_special_tokens=True,
            truncation='only_second',
            max_length=self.max_length
        )
        attention_mask = [1] * len(inputs['input_ids'])
        for tok_idx, tok_id in enumerate(inputs['input_ids']):
            if tok_id == self.source_pad_token_id:
                break
        attention_mask[tok_idx:] = [0] * (self.max_length - tok_idx)
        
        # target sequence
        targets = self.target_tokenizer.encode_plus(
            example['target_answer'],
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length
        )
        targets['decoder_labels'] = targets['input_ids'][1:] + [self.target_tokenizer.pad_token_id]
        decoder_attention_mask = [1] * len(targets['input_ids'])
        for tok_idx, tok_id in enumerate(targets['input_ids']):
            if tok_id == self.target_pad_token_id:
                break
        decoder_attention_mask[tok_idx:] = [0] * (self.max_length - tok_idx)
        
        #
        if 'id' in example:
            id_ = int(example['id'])
        else:
            id_ = -1
        
        #
        return_dict = {
            'input_ids': inputs['input_ids'],
            'attention_mask': attention_mask,
            'decoder_input_ids': targets['input_ids'],
            'decoder_labels': targets['decoder_labels'],
            'decoder_attention_mask': decoder_attention_mask,
            'gold_label': [self.label2id[example['gold_label']]],
            'id': [id_],
        }
        

        return return_dict