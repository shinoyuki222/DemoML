# DemoML - Demonstrations on Machine Learning

Visit [My Blog](https://shinoyuki222.github.io) for more discussion (Mainly in Chinese).

Please leave your comments: [Blog issues](https://github.com/shinoyuki222/shinoyuki222.github.io/issues) or [Demo issues](https://github.com/shinoyuki222/DemoML/issues)

Issues include but *not* limited to:
- Potential topics: any interesting topic which has not been contained.
- Suggestions on coding.
- Share your perspectives
- Raise issues.
- ...
- ...

## Machine Learning Algorithm: Description and Demonstration
### Basics
#### Gaussian Process:
- Demo: [Gaussian Process Demo](https://github.com/shinoyuki222/DemoML/tree/master/Gaussian_Process)
- Understanding: [Gaussian Process 高斯过程](https://shinoyuki222.github.io/2020/01/09/2019-01-09%20Gaussian%20Process/)

#### K-mean:
- Demo: [K-mean Demo](https://github.com/shinoyuki222/DemoML/tree/master/K-mean)
- Understanding: [K-means and Latent variable model](https://shinoyuki222.github.io/2020/04/26/2020-04-26%20K-mean/)

#### Gausian Mixture Model (GMM)
- Demo
- Understanding:

#### Hidden Markov Model (HMM)

#### Conditional Random Field (CRF)


#### Topics not included but can easily be found from other repos:
- Linear Regression
- Logisctic Regression
- Support Vector Machine (SVM)
- Principal Components Analysis (PCA)
- Decision Tree and Random Forest (Bagging)
- Adaboost, Xgboost, GBDT (Boosting)


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

#### <span id="nlu">Natural Language Understanding (NLU): Mixed with Topic classification and NER</span>:
- [Data preparation](https://github.com/shinoyuki222/DemoML/tree/master/NLU/data): additional process needed
- [NLU with Transformer](https://github.com/shinoyuki222/DemoML/tree/master/NLU/main_Transformer)
- [NLU with pretrained BERT](https://github.com/shinoyuki222/DemoML/tree/master/NLU/main_BERT)

#### <span id="nlg">Natural Language Generation (NLG)</span>
- [Data](https://github.com/shinoyuki222/DemoML/tree/master/VAE_NLG/data)
- [NLG using VAE](https://github.com/shinoyuki222/DemoML/tree/master/VAE_NLG)

#### <span id="cb">ChatBot</span>
- [Data](https://github.com/shinoyuki222/DemoML/tree/master/ChatBot/chatbot_data)
- [Seq2Seq ChatBot with LSTM](https://github.com/shinoyuki222/DemoML/tree/master/ChatBot)

#### Relation Extraction
- Bootstrap
- SnowBall

