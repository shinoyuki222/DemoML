# Pytorch NLP - Name Entity Recognition
[Main repo](https://github.com/shinoyuki222/PyTorch_NLP)

Model based on paper:
[Knowledge Graph Embedding Based Question Answering](http://research.baidu.com/Public/uploads/5c1c9a58317b3.pdf)

```
└── shinoyuki222/PyTorch_NLP/tree/master/NER
        |-- README.md
        |-- build_msra_dataset_tags.py
        |-- NER_data
        |   |-- MSRA
        |   |   |-- msra_test_bio
        |   |   |-- msra_train_bio
        |-- main_BERT
        |   |-- data_loader.py
        |   |-- evaluate.py
        |   |-- metrics.py
        |   |-- train.py
        |   |-- utils.py
        |   |-- bert-base-chinese-pytorch
        |   |   |-- bert_config.json
        |   |   |-- pytorch_model.bin
        |   |   |-- vocab.txt
        |   |-- experiments
        |       |-- base_model
        |           |-- evaluate.log
        |           |-- params.json
        |           |-- train.log
        |-- main_LSTM
            |-- consts.py
            |-- dataloader.py
            |-- metric.py
            |-- model.py
            |-- train.py         
```
### Prepared data
  - Make sure you have *NER_data/MSRA/msra_test_bio* and *NER_data/MSRA/msra_train_bio*
  ```shell
  python build_msra_dataset_tags.py
  ```
  will create NER_data/MSRA/train, NER_data/MSRA/dev, NER_data/MSRA/test, respectively.

### To train the LSTM model

#### Train and evaluate your experiment
- Train and test
  ```shell
  cd main_LSTM
  python train.py
  ```
- Test only with best trained model
  ```shell
  cd main_LSTM
  python train.py -xt -lm
  ```
### To train the BERT-pretrained model
#### Get BERT model for PyTorch
- Install [pytorch-pretrained-bert](https://pypi.org/project/pytorch-pretrained-bert/):
    + pip install pytorch-pretrained-bert
- Convert the TensorFlow checkpoint to a PyTorch dump by yourself
    + Download the Google's BERT base model for Chinese from **[`BERT-Base, Chinese`](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)** (Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M parameters), and decompress it.

    + Execute the following command,  convert the TensorFlow checkpoint to a PyTorch dump.

       ```shell
       export TF_BERT_DIR=/PATH_TO/chinese_L-12_H-768_A-12
       export PT_BERT_DIR=/PATH_TO/NER-BERT-pytorch/bert-base-chinese-pytorch
       
       pytorch_pretrained_bert convert_tf_checkpoint_to_pytorch \
       $TF_BERT_DIR/bert_model.ckpt \
       $TF_BERT_DIR/bert_config.json \
       $PT_BERT_DIR/pytorch_model.bin
       ```

    + Copy the BERT parameters file `bert_config.json` and dictionary file `vocab.txt` to the directory `$PT_BERT_DIR`.

       ```shell
       cp $TF_BERT_DIR/bert_config.json $PT_BERT_DIR/bert_config.json
       cp $TF_BERT_DIR/vocab.txt $PT_BERT_DIR/vocab.txt
       ```
#### Train and evaluate your experiment
- if you use default parameters, just run

   ```shell
   cd main_BERT
   python train.py
   ```

   Or specify parameters on the command line
   TBD
<!-- 
   ```shell
   cd main_BERT
   python train.py --data_dir ../NER_data/MSRA --bert_model_dir bert-base-chinese-pytorch --model_dir experiments/base_model --multi_gpu
   ``` -->
- Evaluation on the test set
Once you've run many experiments and selected your best model and hyperparameters based on the performance on the development set, you can finally evaluate the performance of your model on the test set.

   if you use default parameters, just run

   ```shell
   python evaluate.py
   ```

- Prediction
Selected best model and Predict tags.

   ```shell
   python test.py
   ```