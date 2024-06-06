# Assignment 1 - part 1

To run the code in testing mode from the folder path edoardo_cecchinato_247434/LM/part_1:
```console
disi@lab000001:~/edoardo_cecchinato_247434/LM/part_1$ python3 main.py
```

To test different models you can change the parameter *model_type* when call the main function:
```python 
main(train_path, dev_path, test_path, device=DEVICE, model_type="lstm_dropout_adam", is_train=False)
```
The accepted strings are:
* *"rnn_original"*: classic RNN model given by the assignment
* *"lstm"*: classic LSTM model without dropout
* *"lstm_dropout_adam"*: LSTM model with dropout and AdamW as optimizer

If you want to change to training mode, you can change the parameter *is_train* when call the main function:
```python 
main(train_path, dev_path, test_path, device=DEVICE, model_type="lstm_dropout_adam", is_train=False)
```