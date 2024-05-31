from transformers import BertModel
import torch.nn as nn

class BertABSA(nn.Module):
    def __init__(self, model_name, num_aspects):
        super(BertABSA, self).__init__()

        # Load pretrained BERT model
        self.bert = BertModel.from_pretrained(model_name)

        # Aspect term extraction
        self.aspect_out = nn.Linear(self.bert.config.hidden_size, num_aspects)

        # Dropout
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # BERT outputs
        bert_out = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

         # sequence output for aspect term extraction
        sequence_out = bert_out[0]

        # apply dropout
        sequence_out = self.dropout(sequence_out)

        # Aspect term extraction
        aspects = self.aspect_out(sequence_out)
        aspects = aspects.permute(0, 2, 1)

        return aspects