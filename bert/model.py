import torch
from torch import nn
from transformers import BertModel



class BertForQA(nn.Module):
        
    def __init__(self, model_name,layers=1):
        super().__init__()
        
        self.bert = BertModel.from_pretrained(model_name)        
        self.layers = layers
        
        if(self.layers == 1):
            self.qa_outputs = nn.Linear(self.bert.config.hidden_size, 2)
        else:
            intermediate_dim = 256
            self.qa_outputs = nn.Sequential(
                nn.Linear(self.bert.config.hidden_size, intermediate_dim),
                nn.GELU(),
                nn.Dropout(0.4),
                nn.Linear(intermediate_dim, 2)
        )

    def forward(
        self, 
        input_ids,      
        attention_mask=None,  
        token_type_ids=None,  
        start_positions=None,  
        end_positions=None,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs.last_hidden_state
      
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        total_loss = None
        if start_positions is not None and end_positions is not None:
            loss_fn = nn.CrossEntropyLoss()
            total_loss = (loss_fn(start_logits, start_positions) + loss_fn(end_logits, end_positions)) / 2

        return (total_loss, start_logits, end_logits)
