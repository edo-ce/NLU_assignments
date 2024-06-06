# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from utils import *
from model import *
import torch.optim as optim

def main(
        train_path,
        dev_path,
        test_path,
        lr=0.00001,
        clip=5,
        device='cuda:0',
        is_train=False
):

    # retrieve all the data
    data = get_data(train_path, dev_path, test_path)
    lang = data["lang"]

    num_aspects = len(lang.aspect2id)

    # initialize the model
    model_name = "bert-base-uncased"
    model = BertABSA(model_name, num_aspects).to(DEVICE)
    
    path = os.path.join(SAVING_PATH, model.name + ".pt")

    optimizer = optim.Adam(model.parameters(), lr=lr)

    pad_index = -100
    criterion = nn.CrossEntropyLoss(ignore_index=pad_index)

    if not is_train:
        # load saved informations from the bin folder
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        lang = checkpoint["lang"]
        test_data = checkpoint["test_data"]
        print("Loading the model\n")
        print("Evaluating...")
        # run one evaluation and show the results
        results_test, _ = eval_loop(test_data, criterion, model, lang)
        print("Precision: ", results_test[0])
        print("Recall: ", results_test[1])
        print("F1 score: ", results_test[2])
    else:
        print("Training...")
        # train the model on the retrieve data
        train(data, model, optimizer, criterion)


if __name__ == "__main__":
    print(f"Using {DEVICE} device")

    # TODO: remove after debugging
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    
    train_path = os.path.join('..','..','datasets','SemEval_2014_Task_4','laptop14_train.txt')
    dev_path = os.path.join('..','..','datasets','SemEval_2014_Task_4','laptop14_dev.txt')
    test_path = os.path.join('..','..','datasets','SemEval_2014_Task_4','laptop14_test.txt')

    main(train_path, dev_path, test_path, device=DEVICE, is_train=False)