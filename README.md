Some Named Entity Recognition (NER) approaches 
applied to multiple datasets and extended for the main goal of this 
work which is the deidentification of clinical data of electronic
health records.

### Code layout


### References & interesting links to consider :

Stanford NLP-NER tutorial:
https://github.com/cs230-stanford/cs230-code-examples/tree/master/pytorch/nlp

Log-Linear Models, MEMMs, and CRFs:
http://www.cs.columbia.edu/~mcollins/crf.pdf

BiLSTM-CRF:
https://jovian.ai/abdulmajee/bilstm-crf#C29
https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html

Transfer learning - Bert:
https://colab.research.google.com/github/abhimishra91/transformers-tutorials/blob/master/transformers_ner.ipynb

Interesting tutorial, simplistic implementation:
https://www.depends-on-the-definition.com/sequence-tagging-lstm-crf/

Interesting repo, advanced implementation (CONLL):
https://github.com/allanj/pytorch_neural_crf/blob/master/src/model/module/linear_crf_inferencer.py


### Ongoing plan

- [x] Paper reading related to the topic (deidentification)
  - Ch-8 Sequence tagging NLP Book from Jufrasky 
  - other papers and blogposts (see section Reference & interesting links to consider  )
- [x] i2b2 data request
- [x] explore i2b2 dataset
- [x] build code on easier/standard datasets (e.g CoNLL-2003 or [Kaggle NER Dataset](https://www.kaggle.com/datasets/abhinavwalia95/entity-annotated-corpus))
- [ ] lstm, bilstm, then bilstm-crf (consider implementation of pytorch-crf, and crf "from scratch")
- [ ] apply to i2b2 
- [ ] compare to transfer learning (Bert, SciBert, Electra)
- [ ] [FASTER CRF](https://github.com/allanj/pytorch_neural_crf/blob/master/docs/fast_crf.md)

### TODO
Explore : character embedding https://gist.github.com/DuaneNielsen/4e45408948b0aca9b66b7a55ddec8950
          https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html


### Some notes:
#### train : 

activate the environement variables : env-crf (#TODO need to generate install requirements)

```
python train.py --data_dir data/small --model_dir experiments/base_model

```

### more links 
Allennlp (not sure if the code is working)
https://github.com/marumalo/bilstm-crf

NER, multiple papers with code 
http://nlpprogress.com/english/named_entity_recognition.html

An interesting Paper with pseudocode 
https://arxiv.org/pdf/1508.01991.pdf
reference for build upon pytorch tutorial https://github.com/jidasheng/bi-lstm-crf

more codes  :
https://github.com/marumalo/bilstm-crf using AllelNLP

https://github.com/jidasheng/bi-lstm-crf/tree/master/bi_lstm_crf





