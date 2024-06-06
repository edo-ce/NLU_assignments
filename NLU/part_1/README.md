# Assignment 1 - part 1

To run the code in testing mode from the folder path edoardo_cecchinato_247434/NLU/part_1:
```console
disi@lab000001:~/edoardo_cecchinato_247434/NLU/part_1$ python3 main.py
```

To test different models you can change the parameter *model_type* when call the main function:
```python 
main(train_path, dev_path, test_path, device=DEVICE, model_type="bidirectional_dropout", is_train=False)
```
The accepted strings are:
* *"lstm_original"*: original LSTM model without bidirectional and dropout
* *"bidirectional"*: bidirectional LSTM model
* *"bidirectional_dropout"*: bidirectional LSTM model with dropout

If you want to change to training mode, you can change the parameter *is_train* when call the main function:
```python 
main(train_path, dev_path, test_path, device=DEVICE, model_type="lstm_dropout_adam", is_train=False)
```