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
        lr=0.0001,
        clip=5,
        device='cuda:0'
):
    data = get_data(train_path, test_path)
    lang = data["lang"]

    
    num_intents = len(lang.intent2id)
    num_slots = len(lang.slot2id)

    model_name = "bert-base-uncased"
    model = BertIAS(model_name, num_intents, num_slots).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion_slots = nn.CrossEntropyLoss(ignore_index=-100)
    criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token

    train(data, model, optimizer, criterion_slots, criterion_intents, clip=clip)

if __name__ == "__main__":
    print(f"Using {DEVICE} device")

    # TODO: remove after debugging
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    
    train_path = os.path.join('..','..','datasets','ATIS','train.json')
    test_path = os.path.join('..','..','datasets','ATIS','test.json')

    main(train_path, test_path, device=DEVICE)