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
        lr=0.0001,
        clip=5,
        device='cuda:0'
):
    
    data = get_data(train_path, test_path)
    lang = data["lang"]

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)
    
    # model = ModelIAS(hid_size, out_slot, out_int, emb_size, vocab_len, pad_index=PAD_TOKEN).to(device)
    model = ModelIAS_Bidirectional(hid_size, out_slot, out_int, emb_size, vocab_len, pad_index=PAD_TOKEN).to(device)
    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token

    train(data, model, optimizer, criterion_slots, criterion_intents, clip=clip)

if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device} device")

    # TODO: remove after debugging
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    
    train_path = os.path.join('datasets','ATIS','train.json')
    test_path = os.path.join('datasets','ATIS','test.json')

    main(train_path, test_path, device=device)