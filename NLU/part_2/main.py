from functions import *
from model import *
from utils import *
import torch.optim as optim

def main(
        train_path,
        test_path,
        lr=0.0001,
        clip=5,
        device='cuda:0',
        is_train=False
):
    
    # retrieve all the data
    data = get_data(train_path, test_path)
    lang = data["lang"]

    num_intents = len(lang.intent2id)
    print("NUMBER OF INTENTS: ", num_intents)
    num_slots = len(lang.slot2id)
    print("NUMBER OF SLOTS: ", num_slots, '\n')

    # initialize the model
    model_name = "bert-base-uncased"
    model = BertIAS(model_name, num_intents, num_slots).to(device)

    path = os.path.join(SAVING_PATH, model.name + ".pt")

    optimizer = optim.Adam(model.parameters(), lr=lr)

    pad_index = -100
    criterion_slots = nn.CrossEntropyLoss(ignore_index=pad_index)
    criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token

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
        print("Training...")
        # train the model on the retrieve data
        train(data, model, optimizer, criterion_slots, criterion_intents, clip=clip)

if __name__ == "__main__":
    print(f"Using {DEVICE} device")

    # TODO: remove after debugging
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    
    train_path = os.path.join('..','..','datasets','ATIS','train.json')
    test_path = os.path.join('..','..','datasets','ATIS','test.json')

    main(train_path, test_path, device=DEVICE, is_train=False)