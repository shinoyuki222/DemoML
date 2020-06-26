# DemoML
Demos of Machine Learning Algorithm

Please visit [My Blog](https://shinoyuki222.github.io) to read my shallow view of Machine Learning and Natural Language Processing.

## Traditional Machine Learning
### Basic
#### Gaussian Process:
- Demo: [Gaussian Process Demo](https://github.com/shinoyuki222/DemoML/tree/master/Gaussian_Process)
- Understanding: [Gaussian Process 高斯过程](https://shinoyuki222.github.io/2020/01/09/2019-01-09%20Gaussian%20Process/)

#### K-mean:
- Demo: [K-mean Demo](https://github.com/shinoyuki222/DemoML/tree/master/K-mean)
- Understanding: [K-means and Latent variable model](https://shinoyuki222.github.io/2020/04/26/2020-04-26%20K-mean/)

### Deep Learning
#### Transformer:
- PyTorch Demo: [Transformer](https://github.com/shinoyuki222/DemoML/tree/master/Transformer)
- Paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Application: [Sequence tagging task](ner); [Topic classfication](nlu); [Chatbot](https://github.com/shinoyuki222/DemoNLP/tree/master/FreeChat) and other text generation tasks.

#### BERT
- PyTorch Demo: [BERT](https://github.com/shinoyuki222/DemoML/tree/master/BERT)
- Paper: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- Application: [NER](ner) and other sequence tagging task, Topic classfication.

#### Variational AutoEncoder:
- Pytorch Demo: [VAE](https://github.com/shinoyuki222/DemoML/tree/master/VAE_NLG)
- Paper: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- Understanding: [Variational AutoEncoder](https://shinoyuki222.github.io/2020/05/03/2020-05-03%20VAE1/)
- Application: [Natural Language Generation](#nlg)

## Natural language processing
#### Word segmentation: 
- Demo with Description: [Word_segmentation](https://github.com/shinoyuki222/DemoML/tree/master/Word_segmentation)
 
#### <span id="ner">Name Entity Recognition (NER)</span>:
- [Data preparation](https://github.com/shinoyuki222/DemoML/tree/master/NER/NER_data)
- [NER with LSTM](https://github.com/shinoyuki222/DemoML/tree/master/NER/main_LSTM)
- [NER with pretrained BERT](https://github.com/shinoyuki222/DemoML/tree/master/NER/main_BERT)

#### <span id="nlu">Natural Language Understanding: Mixed with Topic classification and NER</span>:
- [Data preparation](https://github.com/shinoyuki222/DemoML/tree/master/NLU/data): additional process needed
- [NLU with Transformer](https://github.com/shinoyuki222/DemoML/tree/master/NLU/main_Transformer)
- [NLU with pretrained BERT](https://github.com/shinoyuki222/DemoML/tree/master/NLU/main_BERT)

#### <span id="nlg">Natural Language Generation(NLG)</span>
- [Data](https://github.com/shinoyuki222/DemoML/tree/master/VAE_NLG/data)
- [NLG using VAE](https://github.com/shinoyuki222/DemoML/tree/master/VAE_NLG)

#### <span id="cb">ChatBot</span>
- [Data](https://github.com/shinoyuki222/DemoML/tree/master/ChatBot/chatbot_data)
- [Seq2Seq ChatBot with LSTM](https://github.com/shinoyuki222/DemoML/tree/master/ChatBot)

