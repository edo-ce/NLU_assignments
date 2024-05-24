import json
import torch
from sklearn.model_selection import train_test_split
from pprint import pprint
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# class to convert labels to ids and vice versa
class Lang():
    def __init__(self, intents, slots):
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents)
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}

    def lab2id(self, elements):
        vocab = {}
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab

# class to define a custom pytorch Dataset
class IntentsAndSlots(Dataset):
    def __init__(self, dataset, lang):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        for x in dataset:
            self.utterances.append(x['utterance'])
            slots = [lang.slot2id[slot] for slot in x['slots'].split()]
            self.slots.append(slots)
            self.intents.append(lang.intent2id[x['intent']])

        self.tokenized_data = self.tokenize(self.utterances, self.slots)
        for i in range(len(self.intents)):
            self.tokenized_data[i]["intents"] = self.intents[i]
            # TODO: check if slot length is correct
            # TODO: see if keep double tensor
            self.tokenized_data[i]["slots_len"] = len(self.slots[i])

        # print(self.tokenized_data[0])

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        return self.tokenized_data[idx]

    def get_label(self, label):
        return self.label_encoder.inverse_transform([label])

    def tokenize(self, utterances, slots):
        res = []
        for utt, slot in zip(utterances, slots):
            utt = utt.split()
            # slot = slot.split()

            tokenized_inputs = self.tokenizer(utt, truncation=True, is_split_into_words=True)
            print(tokenized_inputs)
            raise Exception("STOP HERE")
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

            tokenized_inputs["slots"] = torch.tensor([label_ids])
            res.append(tokenized_inputs)
        return res

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
def collate_fn(batch):
    new_batch = {}
    keys = batch[0].keys()

    for key in keys:
        new_batch[key] = torch.tensor([d[key] for d in batch])

    print(new_batch)

    for key in keys:
        if key != "slots_len":
            new_batch[key] = new_batch[key].to(device)

    return new_batch


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