# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from model import *
from utils import *
import torch.optim as optim

def main(
        train_path,
        test_path,
        hid_size = 200,
        emb_size = 300,
        lr=0.0001, # 0.0001
        clip=5,
        device='cuda:0',
        model_type="bidirectional_dropout",
        is_train=False
):
    
    # retrieve all the data
    data = get_data(train_path, test_path)
    lang = data["lang"]

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    path = os.path.join(SAVING_PATH, model_type + ".pt")

    # initialize the selected model
    if model_type == "lstm_original":
        model = ModelIAS(hid_size, out_slot, out_int, emb_size, vocab_len, pad_index=PAD_TOKEN).to(device)
        print("Using the original LSTM model.\n")
    elif model_type == "bidirectional":
        model = ModelIAS_Bidirectional(hid_size, out_slot, out_int, emb_size, vocab_len, pad_index=PAD_TOKEN, is_dropout=False).to(device)
        print("Using the bidirectional LSTM model.\n")
    else:
        model = ModelIAS_Bidirectional(hid_size, out_slot, out_int, emb_size, vocab_len, pad_index=PAD_TOKEN).to(device)
        print("Using the bidirectional LSTM model with dropout.\n")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()

    if not is_train:
        # load saved informations from the bin folder
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        lang = checkpoint["lang"]
        test_data = checkpoint["test_data"]
        print("Loading the model\n")
        print("Evaluating...")
        # run one evaluation and show the results
        results_test, intent_test, _ = eval_loop(test_data, criterion_slots, criterion_intents, model, lang)
        print('Slot F1: ', results_test['total']['f'])
        print('Intent Accuracy:', intent_test['accuracy'])
    else:
        # initialize the weights of the model
        model.apply(init_weights)
        print("Training...")
        # train the model on the retrieve data
        train(data, model, optimizer, criterion_slots, criterion_intents, clip=clip)

if __name__ == "__main__":
    print(f"Using {DEVICE} device")

    # TODO: remove after debugging
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    
    train_path = os.path.join('..','..','datasets','ATIS','train.json')
    test_path = os.path.join('..','..','datasets','ATIS','test.json')

    main(train_path, test_path, device=DEVICE, model_type="lstm_original", is_train=True)