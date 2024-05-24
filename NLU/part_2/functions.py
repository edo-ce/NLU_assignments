# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import torch
import numpy as np
import torch.optim as optim
from conll import evaluate
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    loss_array = []
    for sample in data:
        print(sample)
        optimizer.zero_grad() # Zeroing the gradient
        slots, intent = model(sample['input_ids'], sample['attention_mask'], sample['token_type_ids'])
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['slots'])
        loss = loss_intent + loss_slot # In joint training we sum the losses.
                                       # Is there another way to do that?
        loss_array.append(loss.item()) # extract loss value as python float
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step() # Update the weights
    return loss_array

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            print(len(sample))
            slots, intents = model(sample['input_ids'], sample['attention_mask'], sample['token_type_ids'])
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['slots'])
            loss = loss_intent + loss_slot
            loss_array.append(loss.item())
            # Intent inference
            # Get the highest probable class
            out_intents = [lang.id2intent[x]
                           for x in torch.argmax(intents, dim=1).tolist()]
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # Slot inference
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                print()
                print(seq)
                length = sample['slots_len']
                print(sample)
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}

    report_intent = classification_report(ref_intents, hyp_intents,
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array

def train(data, model, optimizer, criterion_slots, criterion_intents, clip=5, n_epochs=200, patience=3):
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = 0

    lang = data["lang"]
    
    for x in tqdm(range(1,n_epochs)):
        loss = train_loop(data["train"], optimizer, criterion_slots,
                        criterion_intents, model, clip=clip)
        if x % 5 == 0: # We check the performance every 5 epochs
            sampled_epochs.append(x)
            losses_train.append(np.asarray(loss).mean())
            results_dev, intent_res, loss_dev = eval_loop(data["dev"], criterion_slots,
                                                        criterion_intents, model, lang)
            losses_dev.append(np.asarray(loss_dev).mean())

            f1 = results_dev['total']['f']
            # For decreasing the patience you can also use the average between slot f1 and intent accuracy
            if f1 > best_f1:
                best_f1 = f1
                # Here you should save the model
                patience = 3
            else:
                patience -= 1
            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean

    results_test, intent_test, _ = eval_loop(data["test"], criterion_slots,
                                            criterion_intents, model, lang)
    print('Slot F1: ', results_test['total']['f'])
    print('Intent Accuracy:', intent_test['accuracy'])