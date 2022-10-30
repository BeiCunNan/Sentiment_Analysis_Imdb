# Sentiment_Analysis_Imdb

## Introduction

I use the word2vec、elmo、gpt、fasttext、glove、bert、roberta totally 7 different models and using the
gru、lstm、bilstm、textcnn、rnn totally 5 mothods to run on the imdb datasets. Whitch is so useful for the new boy.

### Dataset

The dataset.csv file is the imdb dataset, which has already been processed. The detailed processing can be found in the
following
article :  [DataPreProcessing](https://beicunnan.blog.csdn.net/article/details/127196715?spm=1001.2014.3001.5502)

In addition to that, I've also covered the process of experimentation in detail on my blog, which you can take a look at
if you're interested Experimenttation
process  [CSDN_IMDB_Sentiment_Analysis](https://blog.csdn.net/ccaoshangfei/article/details/127537953?spm=1001.2014.3001.5501 )

### Network

The network structure is as follows

![Github版 IMDB](https://user-images.githubusercontent.com/105692522/198009720-8bfee092-1a10-41dd-9988-f51ef3ef89cb.png)

### Result

Since IMDB data volume is very large, we use 10% of the data volume for training. The results are as follows

![最终结果](https://user-images.githubusercontent.com/105692522/198865536-3e724bbc-27f1-4656-b27f-25fdbc5b55d3.jpg)



## Requirement

- Python = 3.9
- torch = 1.11.0
- numpy = 1.22.3

## Preparation

### Clone

```bash
git clone https://github.com/BeiCunNan/sentiment_analysis_Imdb.git
```

### Create an anaconda environment

```bash
conda create -n sai python=3.9
conda activate sai
pip install -r requirements.txt
```

## Usage

```bash
python main.py --method sai
```
