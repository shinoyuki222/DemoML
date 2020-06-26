# Name Entity Recognition
[Main repo](https://github.com/shinoyuki222/DemoML/)

### Prepared data
  - Make sure you have *data/corpus_name*
  ```shell
  python process_data.py -l corpus_name
  ```
  will create data/corpus_name, respectively.

### To train the Transformer model
#### Train and evaluate your experiment
Train and test
```shell
cd main_Transformer
python train.py
```
Once you got *Transformer_NER_best.pyt* under your *save_dir*
Evaluate
```shell
cd main_Transformer
python evaluate.py
```
Test
```shell
cd main_Transformer
python test.py
```
#### Extension: Deploying torch model with ONNX
- convert torch model to onnx, test with onnxruntime and simplify onnx model with onnx-simplifier:
```shell
cd main_Transformer
python convert2onnx.py
```
This should helps you got *model_onnx*

- python script as an example to load your exported and simplified onnx model:
```shell
cd main_Transformer
python load_onnx.py
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
       export PT_BERT_DIR=/PATH_TO/bert-base-chinese-pytorch
       
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