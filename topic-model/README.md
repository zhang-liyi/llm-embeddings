## Deep de Finetti: Understanding Why Large Language Models Encode Topic Distributions

Packages used are listed in `requirements.txt`.

Use 

```
pip install -r requirements.txt
```
to install all packages

### Synthetic Datasets

All computations are done in `synthetic-data.ipynb`.

### Natural Corpora

**WikiText-103**

To get data, visit https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/. In this website, click on 'Download WikiText-103 word level' under Download section. Unzip the downloaded .zip file, and put the resulting `wikitext-103/` folder inside the `data/` folder of this repository. Then, the dataset is ready to use.

To get results, run
```
python code/main.py 
  --model_type=[gpt2 / gpt2-medium / gpt2-large / bert / bert-large / llama2 / llama2-chat]
  --dataset=wiki
  --load_embeddings=true
  --token=[0 / 1 / 2]
```

`load_embeddings=true` would compute and save model embeddings for each datapoint; the embeddings correspond to the first, last, and average tokens. `load_embeddings=true` is needed for the first run. For later runs, setting `load_embeddings=false` would save time by automatically loading the previously saved embeddings.

For this dataset in particular, `token=0` refers to using the first token as embedding, `token=1` refers to the last token, and `token=2` refers to average across all tokens in each document.

Additionally, `--learning_rate`, `--batch_size`, and `--weight_decay` arguments can be added, and optimal values are detailed in the paper.

**20Newsgroup**

Dataset is directly acquired from sklearn by running the following code. To get results, run
```
python code/main.py 
  --model_type=[gpt2 / gpt2-medium / gpt2-large / bert / bert-large / llama2 / llama2-chat]
  --dataset=20ng
  --load_embeddings=true
  --token=[1.5]
```

`load_embeddings=true` would compute and save model embeddings for each datapoint; the embeddings correspond to all tokens in each document (differently from WikiText-13, which is larger). `load_embeddings=true` is needed for the first run. For later runs, setting `load_embeddings=false` would save time by automatically loading the previously saved embeddings.

`token` set to anything between 0 and 1 would take the token at `token` percentage into the document. `token` set to 1.5 means taking the average across all tokens in each document.

For both natural corpora datasets, running `results-analysis.ipynb` returns evaluation results, such as the following:

```
bert 0.0003 3.4e-05 2.0
Acc (0.8489687292082501, 0.011446873276171192)
L2 (0.02698141673526523, 0.0002543166502288897)
TV (0.10275460875835406, 0.0009369511918512419)
------
gpt2-large 0.0003 0.0 2.0
Acc (0.8847896440129449, 0.008135796174742135)
L2 (0.023226900432487912, 0.00048764266072404193)
TV (0.09353910938242878, 0.0016413592056654096)
------
```
The first line in each block describes the model and parameter settings. Each other line describes results measured by one metric, with (mean, std).