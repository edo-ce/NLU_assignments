# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from utils import *
from model import *
import os

def main(
        train_path,
        test_path,
        hid_size = 300, # 200 original
        emb_size = 300,
        lr=1.5, # 0.0001
        clip=5,
        device='cuda:0'
):
    data = get_data(train_path, test_path)

if __name__ == "__main__":
    print(f"Using {DEVICE} device")

    # TODO: remove after debugging
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    
    train_path = os.path.join('..','..','datasets','SemEval_2014_Task_4','laptop14_train.txt')
    test_path = os.path.join('..','..','datasets','SemEval_2014_Task_4','laptop14_test.txt')

    main(train_path, test_path, device=DEVICE)