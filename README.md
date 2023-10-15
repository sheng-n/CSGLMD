# Contrastive self-supervised graph convolutional network for detecting the relationship among lncRNAs, miRNAs, and diseases (CSGLMD) BIBM 2023
![figure1](https://github.com/sheng-n/CSGLMD/assets/95516781/f65730e7-e001-4fb2-8b5a-52b46ed4464e)

## BIBM 2023 accept
## 1. Overview
The code for paper "Contrastive self-supervised graph convolutional network for detecting the relationship among lncRNAs, miRNAs, and diseases". The repository is organized as follows:

+ `data/` contains the datasets used in the paper;
+ `code/calculating_similarity.py` is the calculation and integration of lncRNA/miRNA/disease similarities;
+ `code/parms_setting.py`contains hyperparmeters;
+ `code/utils.py` contains preprocessing function of the data;
+ `code/data_preprocess.py` contains the preprocess of data;
+ `code/layer.py` contains contrastive self-supervised GCN layers;
+ `code/instantiation.py` contains CSGLMD model instatiation;
+ `code/train.py` contains training and testing code;
+ `code/main.py` runs CSGLMD;

## 2. Dependencies
* numpy == 1.21.1
* torch == 2.0.0+cu118
* sklearn == 0.24.1
* torch-geometric == 2.3.0

## 3. Quick Start
Here we provide a example to predict the lncRNA-disease association scores under 5-cv1 setting on dataset 1:

1. Download and upzip our data and code files
2. Run main.py (in file-- dataset1/LDA.edgelist, neg_sample-- dataset1/non_LDA.edgelist, validation_type-- 5-cv1, task_type--LDA, feature_type-- normal)

## 4. Contacts
If you have any questions, please email Nan Sheng (shengnan21@mails.jlu.edu.cn)
