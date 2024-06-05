import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
import numpy as np
import os
from tqdm import tqdm
import copy
from utils import DEVICE

SAVING_PATH = os.path.join("..", "..", "bin")

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print("Seed setted.")
    
seed_everything()

def train_loop(data, optimizer, criterion, model, clip=5):
    model.train()
    loss_array = []
    number_of_tokens = []

    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosioning gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step() # Update the weights

    return sum(loss_array)/sum(number_of_tokens)

def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            output = model(sample['source'], training=False)
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])

    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return

def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)

def train(data, model, optimizer, criterion_train, criterion_eval, clip=5, n_epochs=100, patience=3, num_trials=5):
    losses_train = []
    losses_dev = []
    # save logs for non-monotonically triggered AvSGD
    dev_perplexities = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1,n_epochs))
    for epoch in pbar:
        loss = train_loop(data["train"], optimizer, criterion_train, model, clip)
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(data["dev"], criterion_eval, model)
            
            # Non-monotonically Triggered AvSGD
            if isinstance(optimizer, optim.SGD) and 't0' not in optimizer.param_groups[0] and len(dev_perplexities) > num_trials and ppl_dev > min(dev_perplexities[:-num_trials]):
                print("change to AvSGD")
                optimizer = optim.ASGD(model.parameters(), lr=optimizer.param_groups[0]['lr'], t0=0, lambd=0., weight_decay=1e-6)
            dev_perplexities.append(ppl_dev)

            losses_dev.append(np.asarray(loss_dev).mean())
            pbar.set_description("PPL: %f" % ppl_dev)
            if  ppl_dev < best_ppl: # the lower, the better
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            # decrease the patience after that AvSGD is activated
            elif isinstance(optimizer, optim.ASGD):
                patience -= 1
 
            if patience <= 0: # Early stopping with patience
                break

    best_model.to(DEVICE)

    # select what to save
    saving_obj = {
        "model": best_model.state_dict(),
        "test_data": data["test"]
    }

    # save the model in the bin folder
    path = os.path.join(SAVING_PATH, best_model.name + ".pt")
    torch.save(saving_obj, path)

    final_ppl,  _ = eval_loop(data["test"], criterion_eval, best_model)
    print('Test ppl: ', final_ppl)