# Sentiment_Analysis_Imdb

## Introduction

I use the word2vec、elmo、gpt、fasttext、glove、bert、roberta totally 7 different models and using the
gru、lstm、bilstm、textcnn、rnn totally 5 mothods to run on the imdb datasets. Whitch is so useful for the new boy.

### Dataset

The dataset.csv file is the imdb dataset, which has already been processed. The detailed processing can be found in the
following
article :  [DataPreProcessing](https://beicunnan.blog.csdn.net/article/details/127196715?spm=1001.2014.3001.5502)

In addition to that, I've also covered the process of experimentation in detail on my blog, which you can take a look at
if you're interested Experimenttation process

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
