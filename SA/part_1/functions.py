import torch
import numpy as np
import random
import os
from tqdm import tqdm
from evals import evaluate_ote

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
    for sample in data:
        # zeroing the gradients
        optimizer.zero_grad()
        aspects = model(sample['input_ids'], sample['attention_mask'], sample['token_type_ids'])
        loss = criterion(aspects, sample['y_aspects'])
        loss_array.append(loss.item())
        loss.backward()
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        # update the weights
        optimizer.step()
    return loss_array

def eval_loop(data, criterion, model, lang):
    model.eval()
    loss_array = []

    ref_aspects = []
    hyp_aspects = []
    
    with torch.no_grad():
        for sample in data:
            aspects = model(sample['input_ids'], sample['attention_mask'], sample['token_type_ids'])
            loss = criterion(aspects, sample['y_aspects'])
            loss_array.append(loss.item())
            
            # aspect inference
            output_aspects = torch.argmax(aspects, dim=1)
            for id_seq, seq in enumerate(output_aspects):
                gt_ids = sample['y_aspects'][id_seq].tolist()
                
                gt_aspects = []
                out_aspects = []
                
                seq = seq.tolist()
                # skip all the special tokens and the padding tokens
                for id_el, elem in enumerate(gt_ids):
                    if elem != -100:
                        gt_aspects.append(lang.id2aspect[elem])
                        out_aspects.append(lang.id2aspect[seq[id_el]])

                ref_aspects.append(gt_aspects)
                hyp_aspects.append(out_aspects)
                
    # call evaluation function
    results = evaluate_ote(ref_aspects, hyp_aspects)
    
    return results, loss_array

def train(data, model, optimizer, criterion, clip=5, n_epochs=200, patience=3):
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = 0

    lang = data["lang"]
    
    for x in tqdm(range(1,n_epochs)):
        loss = train_loop(data["train"], optimizer, criterion, model, clip=clip)
        if x % 5 == 0: # We check the performance every 5 epochs
            sampled_epochs.append(x)
            losses_train.append(np.asarray(loss).mean())
            results_dev, loss_dev = eval_loop(data["dev"], criterion, model, lang)
            losses_dev.append(np.asarray(loss_dev).mean())

            f1 = results_dev[-1]
            if f1 > best_f1:
                best_f1 = f1
                patience = 3
            else:
                patience -= 1
            if patience <= 0: # Early stopping with patience
                break

    # select what to save
    saving_obj = {
        "model": model.state_dict(),
        "lang": lang,
        "test_data": data["test"]
    }

    # save the model in the bin folder
    path = os.path.join(SAVING_PATH, model.name + ".pt")
    torch.save(saving_obj, path)

    results_test, _ = eval_loop(data["test"], criterion, model, lang)

    print("Precision: ", results_test[0])
    print("Recall: ", results_test[1])
    print("F1 score: ", results_test[2])