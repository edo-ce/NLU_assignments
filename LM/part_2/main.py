from functions import *
from model import *
from utils import *
import torch.optim as optim

def main(
        train_path,
        dev_path,
        test_path,
        hid_size = 300,
        emb_size = 300,
        lr=6.5,
        clip=5,
        device='cuda:0',
        is_train=False
):

    # retrieve all the data
    data = get_data(train_path, dev_path, test_path)
    lang = data["lang"]

    vocab_len = len(lang.word2id)

    # initialize the model
    model = LM_LSTM_REGULARIZED(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)

    path = os.path.join(SAVING_PATH, model.name + ".pt")

    optimizer = optim.SGD(model.parameters(), lr=lr)

    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    if not is_train:
        # load saved informations from the bin folder
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        test_data = checkpoint["test_data"]
        print("Loading the model\n")
        print("Evaluating...")
        # run one evaluation and show the results
        final_ppl,  _ = eval_loop(test_data, criterion_eval, model)
        print('Test ppl: ', final_ppl)
    else:
        print("Training...")
        model.apply(init_weights)
        # train the model on the retrieve data
        train(data, model, optimizer, criterion_train, criterion_eval, clip=clip, n_epochs=100, patience=3)

if __name__ == "__main__":
    print(f"Using {DEVICE} device")

    # TODO: remove after debugging
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    
    train_path = os.path.join('..','..','datasets','PennTreeBank','ptb.train.txt')
    dev_path = os.path.join('..','..','datasets','PennTreeBank','ptb.valid.txt')
    test_path = os.path.join('..','..','datasets','PennTreeBank','ptb.test.txt')

    main(train_path, dev_path, test_path, device=DEVICE, is_train=False)