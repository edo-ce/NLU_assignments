# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from model import *
from utils import *
import torch.optim as optim

def main(
        train_path,
        dev_path,
        test_path,
        hid_size = 300, # 200 original
        emb_size = 300,
        lr=0.0001, # 1.5
        clip=5,
        device='cuda:0'
):
    data = get_data(train_path, dev_path, test_path)
    lang = data["lang"]

    vocab_len = len(lang.word2id)

    # model = LM_RNN(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
    model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
    model.apply(init_weights)

    # optimizer = optim.SGD(model.parameters(), lr=lr)
    # optimizer = optim.AdamW(model.parameters(), lr=lr)
    # Non-monotonically Triggered ASGD
    optimizer = optim.ASGD(model.parameters(), lr=lr, t0=0, lambd=0., weight_decay=1e-6)

    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    train(data, model, optimizer, criterion_train, criterion_eval, clip=clip, n_epochs=100, patience=3)

if __name__ == "__main__":
    print(f"Using {DEVICE} device")

    # TODO: remove after debugging
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    
    train_path = os.path.join('..','..','datasets','PennTreeBank','ptb.train.txt')
    dev_path = os.path.join('..','..','datasets','PennTreeBank','ptb.valid.txt')
    test_path = os.path.join('..','..','datasets','PennTreeBank','ptb.test.txt')

    main(train_path, dev_path, test_path, device=DEVICE)