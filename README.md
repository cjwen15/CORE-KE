# Improving Extraction of Chinese Open Relations Using Pre-trained Language Model and Knowledge Enhancement



## Requirement
* python 3.6.13
* tensorflow-gpu 1.15.0
* numpy 1.19.5

## Download the Pre-trained WoBERT_plus model

Download the pre-trained language model  **chinese_wobert_plus_L-12_H-768_A-12** 
from https://github.com/ZhuiyiTechnology/WoBERT, then move to the folder *bert_model* and *PLM fine-tuning*. 

## Download the LTP model
Download the LTP model *ltp_data_v3.4.0.zip* from http://ltp.ai/download.html .

## Datasets
* The COER dataset is available at https://github.com/TJUNLP/COER .
* The SpanSAOKE dataset is available at https://github.com/Lvzhh/MGD-GNN .

## Input Format
```json
{"sentence":"金州勇士队以101比92战胜了休斯顿火箭队", "label":"B-E1 I-E1 I-E1 I-E1 I-E1 O O O O O O O B-R I-R I-R B-E2 I-E2 I-E2 I-E2 I-E2 I-E2"}
```
## Run

* **follow the readme.md in each folder:**

*1.* Syntactic feature integration

*2.* PLM fine-tuning

*3.* Chinese Open Relation Extraction

```shell script
python main.py
```

