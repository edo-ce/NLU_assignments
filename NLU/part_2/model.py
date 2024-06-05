from transformers import BertModel
import torch.nn as nn

# module to fine-tune BERT for Intent and Slot classification
class BertIAS(nn.Module):
    def __init__(self, model_name, num_intents, num_slots, name="bert_ias"):
        super(BertIAS, self).__init__()

        # Load pretrained BERT model
        self.bert = BertModel.from_pretrained(model_name)

        # Intent classification
        self.intent_out = nn.Linear(self.bert.config.hidden_size, num_intents)

        # Slot filling
        self.slot_out = nn.Linear(self.bert.config.hidden_size, num_slots)

        # Dropout
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)

        self.name = name

    def forward(self, input_ids, attention_mask, token_type_ids):
        # BERT outputs
        bert_out = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # sequence output for slot filling
        sequence_out = bert_out[0]
        # pooled output for intent classification
        pooled_out = bert_out[1]

        # apply dropout
        pooled_out = self.dropout(pooled_out)
        sequence_out = self.dropout(sequence_out)

        # Slot filling
        slots = self.slot_out(sequence_out)
        slots = slots.permute(0,2,1)

        # Intent classification
        intent = self.intent_out(pooled_out)

        return slots, intent