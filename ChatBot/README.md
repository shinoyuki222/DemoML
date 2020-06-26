# Pytorch NLP - FreeChat
[Main repo](https://github.com/shinoyuki222/PyTorch_NLP)


```
└── shinoyuki222/PyTorch_NLP/tree/master/FreeChat
    |-- README.md
    |-- pt_seq2seq_atten_train.py
    |-- chatbot_data
    |   |-- core_reduced
    |-- save
        |-- chatbot_model
            |-- core_reduced
                |-- 2-2_500
                    |-- 3500_checkpoint.tar
                    |-- 4000_checkpoint.tar      
```
#### To train the model
    python pt_seq2seq_atten_train.py

#### To load pretrained model and evaluate 

    python pt_seq2seq_atten_train.py -l -cp 4000 -xt

#### Arguments help
    python pt_seq2seq_atten_train.py -h
