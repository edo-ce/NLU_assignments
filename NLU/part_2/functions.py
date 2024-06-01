# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import torch
import numpy as np
import os
from conll import evaluate
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm

# function used to train the model
def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    loss_array = []
    for sample in data:
        # zeroing the gradients
        optimizer.zero_grad()
        slots, intent = model(sample['input_ids'], sample['attention_mask'], sample['token_type_ids'])
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        loss = loss_intent + loss_slot
        loss_array.append(loss.item())
        loss.backward()
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        # update the weights
        optimizer.step()
    return loss_array

# function used to test the model
def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []
    
    # avoid the creation of computational graph
    with torch.no_grad():
        for sample in data:
            slots, intents = model(sample['input_ids'], sample['attention_mask'], sample['token_type_ids'])
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
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
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterance'][id_seq][:length]
                gt_ids = sample['y_slots'][id_seq].tolist()
                
                gt_slots = []
                to_decode = []
                seq = seq.tolist()
                # skip all the special tokens and the padding tokens
                for id_el, elem in enumerate(gt_ids):
                    if elem != -100:
                        gt_slots.append(lang.id2slot[elem])
                        to_decode.append(seq[id_el])

                utterance = lang.untokenize(utt_ids).split()
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

def plot_train_dev_loss(sampled_epochs, losses_train, losses_dev):
    plt.figure(num = 3, figsize=(8, 5)).patch.set_facecolor('white')
    plt.title('Train and Dev Losses')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.plot(sampled_epochs, losses_train, label='Train loss')
    plt.plot(sampled_epochs, losses_dev, label='Dev loss')
    plt.legend()
    plt.show()

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
            if f1 > best_f1:
                best_f1 = f1
                # Here you should save the model
                patience = 3
            else:
                patience -= 1
            if patience <= 0: # Early stopping with patience
                break

    results_test, intent_test, _ = eval_loop(data["test"], criterion_slots,
                                            criterion_intents, model, lang)
    print('Slot F1: ', results_test['total']['f'])
    print('Intent Accuracy:', intent_test['accuracy'])

    plot_train_dev_loss(sampled_epochs, losses_train, losses_dev)

def save_model(model_name, obj):
    PATH = os.path.join("bin", model_name)
    saving_object = obj
    torch.save(saving_object, PATH)