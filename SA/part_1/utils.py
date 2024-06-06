import re
import random
import torch
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

PAD_TOKEN = 0
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# class to convert labels to ids and vice versa
class Lang():
    def __init__(self, aspects):
        self.aspect2id = self.lab2id(aspects)
        self.id2aspect = {v:k for k, v in self.aspect2id.items()}

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def lab2id(self, elements):
        vocab = {}
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab
    
    def tokenize(self, words, aspects):
        res = []
        for wrd, asp in zip(words, aspects):

            tokenized_inputs = self.tokenizer(wrd, truncation=True, is_split_into_words=True)

            # padding and special tokens are None
            word_ids = tokenized_inputs.word_ids(batch_index=0)
            previous_word_idx = None
            aspect_ids = []
            # Set padding and special tokens to -100.
            for word_idx in word_ids:
                if word_idx is None:
                    aspect_ids.append(-100)
                elif word_idx != previous_word_idx:
                    aspect_ids.append(asp[word_idx])
                else:
                    aspect_ids.append(-100)
                previous_word_idx = word_idx

            tokenized_inputs["aspects"] = aspect_ids
            res.append(tokenized_inputs)
        return res
    
    # function to come back to words given the token ids
    def untokenize(self, token_ids):
        # inverse process of the tokenizer
        decoded = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        # split in 2 every word like i'd for every possible letters
        return re.sub(r"([a-zA-Z])(')([a-zA-Z])", r"\1 \2\3", decoded)
    
class AspectTerms(Dataset):
    def __init__(self, dataset, lang):
        self.words = []
        self.aspects = []

        for x in dataset:
            self.words.append(x['words'])
            aspects = [lang.aspect2id[aspect] for aspect in x['aspects']]
            self.aspects.append(aspects)

        self.tokenized_data = lang.tokenize(self.words, self.aspects)

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        item = self.tokenized_data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"]),
            "attention_mask": torch.tensor(item["attention_mask"]),
            "token_type_ids": torch.tensor(item["token_type_ids"]),
            "aspects": torch.tensor(item["aspects"])
        }

# function to retrieve data from the file
def read_file(path):
    dataset = []
    words = []
    aspects = []

    with open(path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                # split the line into word and tag
                word, aspect = line.split()
                words.append(word)
                aspects.append(aspect.split('-')[0])
            elif len(words) > 0:
                dataset.append({"words": words, "aspects": aspects})
                words = []
                aspects = []
    
    return dataset

# function to retrieve the aspects
def get_aspects(train, dev, test):
    aspects = set()
    for data in (train + dev + test):
        for aspect in data["aspects"]:
            aspects.add(aspect)
    return aspects

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

    src_wrd, _ = merge(new_item['input_ids'])
    src_mask, _ = merge(new_item['attention_mask'])
    src_token_ids, _ = merge(new_item['token_type_ids'])
    y_aspects, y_lengths = merge(new_item["aspects"], pad_token=-100)

    src_wrd = src_wrd.to(DEVICE) # We load the Tensor on our selected device
    src_mask = src_mask.to(DEVICE)
    src_token_ids = src_token_ids.to(DEVICE)
    y_aspects = y_aspects.to(DEVICE)
    y_lengths = torch.LongTensor(y_lengths).to(DEVICE)

    # new_item["words"] = new_item["input_ids"]
    # 2-dimensional vector (examples x max_number of tokens in the batch)
    new_item["input_ids"] = src_wrd
    new_item["attention_mask"] = src_mask
    new_item["token_type_ids"] = src_token_ids
    new_item["y_aspects"] = y_aspects
    new_item["aspects_len"] = y_lengths

    return new_item

def get_data(train_path, dev_path, test_path):
    train_raw = read_file(train_path)
    dev_raw = read_file(dev_path)
    test_raw = read_file(test_path)

    print('TRAIN size:', len(train_raw))
    print('DEV size:', len(dev_raw))
    print('TEST size:', len(test_raw), '\n')

    aspects = get_aspects(train_raw, dev_raw, test_raw)

    lang = Lang(aspects)

    train_dataset = AspectTerms(train_raw, lang)
    dev_dataset = AspectTerms(dev_raw, lang)
    test_dataset = AspectTerms(test_raw, lang)

    # Dataloader instantiation
    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    return {
        "train": train_loader, 
        "dev": dev_loader, 
        "test": test_loader, 
        "lang": lang
    }