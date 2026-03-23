# LLM Embeddings
Code for Paper - What Should Embeddings Embed? Autoregressive Models Represent Latent Generating Distributions

Packages used are listed in `requirements.txt`.

Use

```
pip install -r requirements.txt
```
to install all packages.

## Topic Model Experiments

Navigate to the `topic-model` folder.

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

`load_embeddings=true` would compute and save model embeddings for each datapoint; the embeddings correspond to all tokens in each document (differently from WikiText-103, which is larger). `load_embeddings=true` is needed for the first run. For later runs, setting `load_embeddings=false` would save time by automatically loading the previously saved embeddings.

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

## HMM Experiments

Navigate to the rest-of-dist folder of the repository.

Data is generated from a Hidden Markov Model with 4 hidden states. The transition matrix is drawn from a symmetric Dirichlet prior, and each row of the emission matrix is drawn from a Dirichlet prior with concentration `delta`. A transformer is trained autoregressively on the generated token sequences.

After training, probing is automatically run for three inference targets:
- `viterbi`: the most likely hidden state at the last observed position (Viterbi decoding)
- `posterior`: the predictive distribution over the next hidden state p(z_{n+1} | x_{1:n})
- `posterior_np1`: the filtering posterior over the last hidden state p(z_{n+1} | x_{1:n+1})

To run an experiment, use
```
python main.py
  --model_name=hmm
  --V=[vocab_size]
  --T=[sequence_length]
  --delta=[emission_concentration]
  --N=[num_training_sequences]
  --seed=[seed]
```

`--V` sets the vocabulary size (number of distinct emission symbols). `--T` sets the sequence length. `--delta` is the Dirichlet concentration parameter for the emission distribution; smaller values produce more peaked, state-distinctive emissions. `--N` sets the number of training sequences.

For each inference target, probing is run with `token_unseen=0` (the embedding is extracted from the full sequence of length T) and `token_unseen=1` (the last token is withheld, so the embedding is extracted after seeing T-1 tokens). This tests whether the embedding reflects the updated belief after observing the final token.

Optionally, `--mlm=true` trains the transformer with masked language modeling instead of autoregressive next-token prediction.

Results are saved to `results/` as CSV files with the naming convention
`hmm_default_{target}_{token_unseen}_{delta}_{lr}_{T}_{seed}.csv`,
containing the cross-entropy loss, L2 loss, and accuracy of the linear probe on the test set.

## Exchangeable Distribution Experiments

Navigate to the root folder of the repository.

Data is generated from conjugate Bayesian models in which observations are exchangeable given a latent parameter. Three models are supported: `gaussian-gamma` (normal likelihood with unknown mean and precision), `bernoulli` (Bernoulli likelihood with Beta prior), and `exponential` (exponential likelihood with Gamma prior). A transformer is trained autoregressively on the generated sequences, and a linear probe is trained on its embeddings to recover the sufficient statistics or posterior moments of the latent parameter.

An out-of-distribution (OOD) evaluation is automatically run after the in-distribution probe, using data generated from a different set of hyperparameters.

**Gaussian-Gamma**

To get results, run
```
python main.py
  --model_name=gaussian-gamma
  --target_name=[sufficient_stat / moments]
  --T=[sequence_length]
  --N=[num_training_sequences]
  --token_num=[token_index]
  --generate_data=[true / false]
```

`generate_data=true` generates and saves a new dataset; it is needed for the first run. For later runs, setting `generate_data=false` loads the previously saved dataset and saves time.

`--token_num` selects which token position's embedding is extracted from the transformer (0-indexed). For example, `--token_num=498` uses the embedding at position 499 in a sequence of length 500, i.e., after seeing almost the full context. Setting `--token_num=0.5` averages embeddings across all positions.

`--target_name=sufficient_stat` probes for the sample mean and sample standard deviation of the observed sequence, which are the sufficient statistics of the Gaussian-Gamma model. `--target_name=moments` probes for four moments of the posterior predictive distribution.

In-distribution hyperparameters are `[mu_0=1, lambda_0=1, alpha=5, beta=1]`; OOD hyperparameters are `[mu_0=5, lambda_0=1, alpha=2, beta=1]`. Both evaluations are run automatically.

Additionally, `--mlm=true` trains with masked language modeling instead of autoregressive prediction, and `--use_linear_classifier=false` switches the probe from linear to a two-layer MLP.

Results are saved to `results/` as CSV files. In-distribution and OOD results are saved separately with `pred_results_` and `pred_results_ood_` prefixes, along with the corresponding true targets and validation losses.



