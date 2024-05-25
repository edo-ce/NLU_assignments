import json
import torch
from sklearn.model_selection import train_test_split
from pprint import pprint
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

PAD_TOKEN = 0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# class to convert labels to ids and vice versa
class Lang():
    def __init__(self, intents, slots):
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents)
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def lab2id(self, elements):
        vocab = {}
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab
    
    def tokenize(self, utterances, slots):
        res = []
        for utt, slot in zip(utterances, slots):
            utt = utt.split()
            # slot = slot.split()

            tokenized_inputs = self.tokenizer(utt, truncation=True, is_split_into_words=True)
            # tokenized_inputs = self.tokenizer(utt, truncation=True, is_split_into_words=True, padding="max_length", max_length=512, return_tensors="pt")

            # padding and special tokens are None
            word_ids = tokenized_inputs.word_ids(batch_index=0)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set padding and special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(slot[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            tokenized_inputs["slots"] = label_ids
            res.append(tokenized_inputs)
        return res
    
    def untokenize(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

# class to define a custom pytorch Dataset
class IntentsAndSlots(Dataset):
    def __init__(self, dataset, lang):
        self.utterances = []
        self.intents = []
        self.slots = []

        for x in dataset:
            self.utterances.append(x['utterance'])
            slots = [lang.slot2id[slot] for slot in x['slots'].split()]
            self.slots.append(slots)
            self.intents.append(lang.intent2id[x['intent']])

        self.tokenized_data = lang.tokenize(self.utterances, self.slots)
        for i in range(len(self.intents)):
            self.tokenized_data[i]["intent"] = self.intents[i]

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        item = self.tokenized_data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"]),
            "attention_mask": torch.tensor(item["attention_mask"]),
            "token_type_ids": torch.tensor(item["token_type_ids"]),
            "slots": torch.tensor(item["slots"]),
            "intent": item["intent"]
        }

# function to load data
def load_data(path):
    '''
        input: path/to/data
        output: json
    '''
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset

def get_intents(train, test):
    intents = set()
    for data in (train + test):
        intents.add(data["intent"])
    return intents

def get_slots(train, test):
    slots = set()
    for data in (train + test):
        for slot in data["slots"].split():
            slots.add(slot)
    return slots

# function to split train data into train and dev
def split_test(tmp_train, portion=0.1):
    intents = [x['intent'] for x in tmp_train]
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1: # If some intents occurs only once, we put them in training
            inputs.append(tmp_train[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train[id_y])
    # Random Stratify
    X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion,
                                                        random_state=42,
                                                        shuffle=True,
                                                        stratify=labels)
    X_train.extend(mini_train)
    
    return X_train, X_dev

# function to use for the pytorch Dataloader collate_fn
def collate_fn(data):
    def merge(sequences, pad_token=PAD_TOKEN):
        '''
        merge from batch * sent_len to batch * max_len
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        # print(padded_seqs)
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths
    # Sort data by seq lengths
    data.sort(key=lambda x: len(x['input_ids']), reverse=True)
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['input_ids'])
    src_mask, _ = merge(new_item['attention_mask'])
    src_token_ids, _ = merge(new_item['token_type_ids'])
    # TODO: check if it works correctly
    y_slots, y_lengths = merge(new_item["slots"], pad_token=-100)
    intent = torch.LongTensor(new_item["intent"])

    src_utt = src_utt.to(DEVICE) # We load the Tensor on our selected device
    src_mask = src_mask.to(DEVICE)
    src_token_ids = src_token_ids.to(DEVICE)
    y_slots = y_slots.to(DEVICE)
    intent = intent.to(DEVICE)
    y_lengths = torch.LongTensor(y_lengths).to(DEVICE)

    new_item["utterance"] = new_item["input_ids"]
    # 2-dimensional vector (examples x max_number of tokens in the batch)
    new_item["input_ids"] = src_utt
    new_item["attention_mask"] = src_mask
    new_item["token_type_ids"] = src_token_ids
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths

    return new_item


def get_data(train_path, test_path):
    tmp_train_raw = load_data(train_path)
    test_raw = load_data(test_path)

    print('Train samples:', len(tmp_train_raw))
    print('Test samples:', len(test_raw))
    pprint(tmp_train_raw[0])
    print()

    intents = get_intents(tmp_train_raw, test_raw)
    slots = get_slots(tmp_train_raw, test_raw)

    lang = Lang(intents, slots)

    train_raw, dev_raw = split_test(tmp_train_raw)
    print('Train data has been splitted')
    print('TRAIN size:', len(train_raw))
    print('DEV size:', len(dev_raw))
    print('TEST size:', len(test_raw), '\n')

    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    return {"train": train_loader, 
            "dev": dev_loader, 
            "test": test_loader, 
            "lang": lang}