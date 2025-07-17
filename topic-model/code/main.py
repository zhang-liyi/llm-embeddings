import os

import torch

import nltk
nltk.download('stopwords')
nltk.download('punkt')

from util.data_processing import *
from util.lda import *
import util.classifier_training as classifier_training
import util.get_model as get_model
from public_version.code.parse_args import *

args = parse_args()
config = vars(args)
load = args.load_embeddings

job = args.job.lower()
use_linear_classifier = args.use_linear_classifier
train_mode = args.train_mode.lower()
token = args.token
use_wandb = args.use_wandb

# -------------- LLM --------------+
##### Get model and tokenizer
model, tokenizer = get_model.get_model(config)

device = 'cuda'
if 'gpt2' in config['model_type'] or 'bert' in config['model_type']:
    model = model.to(torch.device(device))
if config['load_finetune'] != 'none':
    model.load_state_dict(torch.load(config['load_finetune'] + 'model_weights.pth'))

# ------------- For different datasets ---------+
if config['dataset'] == '20ng':

    # --------- Data processing ---------+
    lda_docs, init_docs_t, init_docs_v = load_20ng(config)
    N_t = len(init_docs_t)
    N_v = len(init_docs_v)
    print(init_docs_t[0:10])

    # --------- LDA x Gensim --------+
    lda_models, lda_corpus_t, lda_corpus_v = get_lda_models(lda_docs, N_t)

    # --------- Run Method ------------+
    ##### Get model embeddings
    # Takes a couple of hours if not load
    if not load:
        N_t = get_embeddings_20ng(config, init_docs_t, model, tokenizer, lda_models, lda_corpus_t, train_val='train', device=device)
        N_v = get_embeddings_20ng(config, init_docs_v, model, tokenizer, lda_models, lda_corpus_v, train_val='val', device=device)

    target_mixtures_t, model_outputs_t, losses_t = load_embeddings(config, N_t, train_val='train')
    target_mixtures_v, model_outputs_v, losses_v = load_embeddings(config, N_v, train_val='val')

    print('Length', len(lda_corpus_t), len(model_outputs_t), len(losses_t), len(target_mixtures_t), 
        len(lda_corpus_v), len(model_outputs_v), len(losses_v), len(target_mixtures_v))

    # Delete items in lda corpus that exceed length 1024
    max_len = 1024 if 'bert' not in config['model_type'] else 512
    config['max_len'] = max_len
    del_indices = []
    for i in range(len(init_docs_t)):
        with torch.no_grad():
            input = tokenizer(init_docs_t[i], return_tensors='pt')
            if input['input_ids'].shape[1] > max_len:
                del_indices.append(i)
    del_indices.sort(reverse=True)
    for i in del_indices:
        del lda_corpus_t[i]
        if job == 'finetuning':
            del init_docs_t[i]

    del_indices = []
    for i in range(len(init_docs_v)):
        with torch.no_grad():
            input = tokenizer(init_docs_v[i], return_tensors='pt')
            if input['input_ids'].shape[1] > max_len:
                del_indices.append(i)
    del_indices.sort(reverse=True)
    for i in del_indices:
        del lda_corpus_v[i]
        if job == 'finetuning':
            del init_docs_v[i]


elif config['dataset'] == 'wiki':

    dataset_by_article_t, dataset_ids_t, lda_docs_t = load_wiki(tokenizer, 'train')
    dataset_by_article_v, dataset_ids_v, lda_docs_v = load_wiki(tokenizer, 'val')
    N_t = len(dataset_ids_t)
    N_v = len(dataset_ids_v)
    lda_docs = lda_docs_t + lda_docs_v
    lda_models, lda_corpus_t, lda_corpus_v = get_lda_models(lda_docs, N_t, passes=3, debug=False)

    if not load:
        N_t = get_embeddings_wiki(config, dataset_ids_t, lda_corpus_t, model, tokenizer, lda_models, train_val='train', device=device)
        N_v = get_embeddings_wiki(config, dataset_ids_v, lda_corpus_v, model, tokenizer, lda_models, train_val='val', device=device)

    if not config['pos']:
        target_mixtures_t, model_outputs_t, losses_t = load_embeddings(config, N_t, train_val='train')
        target_mixtures_v, model_outputs_v, losses_v = load_embeddings(config, N_v, train_val='val')

        print('Length', len(lda_corpus_t), len(model_outputs_t), len(losses_t), len(target_mixtures_t), 
            len(lda_corpus_v), len(model_outputs_v), len(losses_v), len(target_mixtures_v))
        
    else:
        target_mixtures_noun_t, target_mixtures_verb_t, target_mixtures_adp_t, \
            model_outputs_noun_t, model_outputs_verb_t, model_outputs_adp_t = load_embeddings(config, N_t, train_val='train')
        target_mixtures_noun_v, target_mixtures_verb_v, target_mixtures_adp_v, \
            model_outputs_noun_v, model_outputs_verb_v, model_outputs_adp_v = load_embeddings(config, N_v, train_val='val')

##### Training
if job == 'train_classifier':
    if config['pos'] == False:
        classifier_training.train(config, lda_models, target_mixtures_t, target_mixtures_v, model_outputs_t, model_outputs_v, lda_corpus_t, lda_corpus_v, token, device)
    else:
        classifier_training.train(config, lda_models, target_mixtures_noun_t, target_mixtures_noun_v, 
                                    model_outputs_noun_t, model_outputs_noun_v, lda_corpus_t, lda_corpus_v, token, device, save_appendix='_noun')
        classifier_training.train(config, lda_models, target_mixtures_verb_t, target_mixtures_verb_v, 
                                    model_outputs_verb_t, model_outputs_verb_v, lda_corpus_t, lda_corpus_v, token, device, save_appendix='_verb')
        classifier_training.train(config, lda_models, target_mixtures_adp_t, target_mixtures_adp_v, 
                                model_outputs_adp_t, model_outputs_adp_v, lda_corpus_t, lda_corpus_v, token, device, save_appendix='_adp')
if job == 'finetuning':
    #finetuning.train(config, init_docs_t, init_docs_v, model, tokenizer, device)
    pass

