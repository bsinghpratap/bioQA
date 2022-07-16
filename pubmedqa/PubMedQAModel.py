import torch
import torch.nn as nn
from transformers import (BertPreTrainedModel, BertModel, AdamW, get_linear_schedule_with_warmup,
                          RobertaPreTrainedModel, RobertaModel,
                          AutoTokenizer, AutoModel, AutoConfig)
from transformers import (WEIGHTS_NAME,
                          AutoModelForSequenceClassification,
                          BertConfig, BertForSequenceClassification, BertTokenizer,
                          XLMConfig, XLMForSequenceClassification, XLMTokenizer,
                          DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
import numpy as np
import os
import json
import utility_fns

# model class
class QAModel(nn.Module):
    def __init__(
            self,
            model_name,
            num_classes,
    ):
        super(QAModel, self).__init__()

        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_classes,
            finetuning_task='pubmedqa'
        )
        self.encoder = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
        )

        return

    def forward(
            self,
            batch_,
    ):
        outputs = self.encoder(**batch_)
        logits_ = outputs[0]

        return logits_