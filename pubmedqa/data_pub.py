import torch
import random
from torch.utils.data import Dataset
import scipy.sparse
from pdb import set_trace as breakpoint
from collections import Counter
from random import shuffle

class pubmedDataset(Dataset):
    """Class to load the dataset and get batches of paras"""
    
    def __init__(self, list_data, 
                 tokenizer, max_length, label2id):
        
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
        inputs = self.tokenizer.encode_plus(example['sentence1'],
                                            example['sentence2'],
                                            add_special_tokens=True,
                                            truncation=True,
                                            max_length=self.max_length)
                
        input_ids = inputs["input_ids"]
        input_ids = input_ids[:self.max_length]
#         token_type_ids = token_type_ids[:self.max_length]
        attention_mask = [1] * len(input_ids)
        
        padding_length = self.max_length - len(input_ids)
        input_ids = input_ids + ([self.pad_token] * padding_length)
#         token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        
        assert len(input_ids) == self.max_length, "Error with input length {} vs {}".format(len(input_ids), self.max_length)
        assert len(attention_mask) == self.max_length, "Error with input length {} vs {}".format(len(attention_mask), self.max_length)
#         assert len(token_type_ids) == self.max_length, "Error with input length {} vs {}".format(len(token_type_ids), self.max_length)
        
        label = self.label2id[example['gold_label']]
        return_dict = {'input_ids':torch.LongTensor(input_ids),
                       'attention_mask':torch.LongTensor(attention_mask),
                       'label': torch.LongTensor([label])}

        
#         return_dict = {'input_ids':torch.LongTensor(input_ids),
#                        'attention_mask':torch.LongTensor(attention_mask),
#                        'token_type_ids':torch.LongTensor(token_type_ids),
#                        'label': torch.LongTensor([label])}
                
        return return_dict