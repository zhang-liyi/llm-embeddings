import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import re
import string
import torch
import torch.distributions as D
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model
from sklearn.datasets import fetch_20newsgroups
import spacy
import json
import datetime
import os

from nltk.tokenize import word_tokenize

def has_punctuation(w):
    return any(char in string.punctuation for char in w)

def has_number(w):
    return any(char.isdigit() for char in w)

def load_20ng(config):
    remove_header = config['remove_header']
    use_eos_token = config['use_eos_token']
    raw = config['raw']
    # Read data
    print('-- Reading data --')
    train_data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    test_data = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

    docs_t = [word_tokenize(train_data.data[i]) for i in range(len(train_data.data))]
    docs_v = [word_tokenize(train_data.data[i]) for i in range(len(test_data.data))]

    lda_docs = docs_t + docs_v
    lda_docs = [[w.lower() for w in lda_docs[i] if not has_punctuation(w)] for i in range(len(lda_docs))]
    lda_docs = [[w for w in lda_docs[i] if not has_number(w)] for i in range(len(lda_docs))]
    lda_docs = [[w for w in lda_docs[i] if len(w)>1] for i in range(len(lda_docs))]
    lda_docs = [" ".join(lda_docs[i]) for i in range(len(lda_docs))]

    docs_t = []
    for i, doc in enumerate(train_data['data']):
        if use_eos_token:
            docs_t.append("<|endoftext|> ")
        else:
            docs_t.append("")
        word_list = doc.split(' ')

        if not raw:
            for word in word_list:
                if '---' in word or '===' in word or '___' in word or len(word.strip()) == 0:
                    continue
                elif '\n' in word:
                    word = re.sub('\s+',' ',word)
                docs_t[i] += word + ' '
            docs_t[i] = docs_t[i][:-1]
            if use_eos_token:
                docs_t[i] += "<|endoftext|>"
        else:
            for word in word_list:
                docs_t[i] += word + ' '
        docs_t[i] = docs_t[i][:-1]

    docs_v = []
    for i, doc in enumerate(test_data['data']):
        if use_eos_token:
            docs_v.append("<|endoftext|> ")
        else:
            docs_v.append("")
        word_list = doc.split(' ')

        if not raw:
            for word in word_list:
                if '---' in word or '===' in word or '___' in word or len(word.strip()) == 0:
                    continue
                elif '\n' in word:
                    word = re.sub('\s+',' ',word)
                docs_v[i] += word + ' '
            docs_v[i] = docs_v[i][:-1]
            if use_eos_token:
                docs_v[i] += "<|endoftext|>"
        else:
            for word in word_list:
                docs_v[i] += word + ' '
        docs_v[i] = docs_v[i][:-1]

    return lda_docs, docs_t, docs_v


def load_wiki(tokenizer, train_val='train', path=''):

    max_len = 512

    if train_val == 'train':
        dataset_file = f"{path}data/wikitext-103/wiki.train.tokens"
    if train_val == 'val':
        dataset_file = f"{path}data/wikitext-103/wiki.valid.tokens"
    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset = f.readlines()

    dataset_by_article = []
    count = -1
    for line in dataset[1:]:
        if line[:3] == ' = ' and line[3:5] != '= ':
            count += 1
            dataset_by_article.append(line)
        else:
            dataset_by_article[count] += line

    dataset_ids = []
    for i, article in enumerate(dataset_by_article):
        article_ids = tokenizer(article, return_tensors='pt')['input_ids'][0]
        num_sequences = len(article_ids) // max_len
        mod = len(article_ids) % max_len
        if len(article_ids) < 50:
            continue
        split_sizes = [max_len for _ in range(num_sequences)]
        if mod != 0:
            split_sizes.append(mod)
        article_ids = torch.split(article_ids, split_sizes, 0)
        # if len(article_ids[-1]) < 50:
        #     article_ids = article_ids[:-1]
        dataset_ids.extend(list(article_ids))

    lda_docs = tokenizer.batch_decode(dataset_ids)
    lda_docs = [word_tokenize(lda_docs[i]) for i in range(len(lda_docs))]

    lda_docs = [[w.lower() for w in lda_docs[i] if not has_punctuation(w)] for i in range(len(lda_docs))]
    lda_docs = [[w for w in lda_docs[i] if not has_number(w)] for i in range(len(lda_docs))]
    lda_docs = [[w for w in lda_docs[i] if len(w)>1] for i in range(len(lda_docs))]
    lda_docs = [" ".join(lda_docs[i]) for i in range(len(lda_docs))]
    
    return dataset_by_article, dataset_ids, lda_docs
def get_target_only_20ng(config, init_docs, tokenizer, lda_models, lda_corpus, K=20, train_val='train', device='cpu'):

    remove_header = config['remove_header']
    use_eos_token = config['use_eos_token']
    raw = config['raw']
    model_outputs = []
    target_mixtures = [[],[],[]]
    header_option = '_noheader' if remove_header else ''
    eos_option = '_eos' if use_eos_token else ''
    raw_option = '_raw' if raw else ''
    finetune = '_ft' if config['load_finetune'] != 'none' else ''
    full_len = '_full' if config['token'] == 2.5 else ''
    save_every = 200 if config['token'] == 2.5 else 1000
    model_type = config['model_type']
    embd_layer = config['embd_layer']
    embd_layer_int = config['embd_layer_int']

    max_len = 1024 if 'bert' not in model_type else 512

    N = len(init_docs)
    print('length', N)
    for i in range(N):
        with torch.no_grad():
            if config['token'] != 2.5:
                input = tokenizer(init_docs[i], return_tensors='pt').to(torch.device(device))
            else:
                input = tokenizer(init_docs[i], return_tensors='pt', padding='max_length').to(torch.device(device))
            if input['input_ids'].shape[1] <= max_len and input['input_ids'].shape[1] > 1:
                input_ids = input.input_ids
                for lda_idx in range(3):
                    mixtures = lda_models[lda_idx].get_document_topics(lda_corpus[i], minimum_probability=1e-15)
                    mixtures = [tup[1] for tup in mixtures]
                    target_mixtures[lda_idx].append(mixtures)
            if i+1 == N:
                
                with open(f'data/20ng-targetmix{header_option}{eos_option}{raw_option}{finetune}{full_len}_{model_type}_{embd_layer}_K{K}_{train_val}.pickle', 'wb') as handle:
                    pickle.dump(target_mixtures, handle, protocol=pickle.HIGHEST_PROTOCOL)
        torch.cuda.empty_cache()

    return N


def get_embeddings_wiki(config, dataset_ids, lda_corpus, model, tokenizer, lda_models, K=20, train_val='train', device='cpu'):

    model_type = config['model_type']
    embd_layer = config['embd_layer']
    embd_layer_int = config['embd_layer_int']
    model_outputs = []
    target_mixtures = [[],[],[]]
    N = len(dataset_ids)
    print('length', N)
    model.eval()
    save_every = 10000
    if config['load_embeddings_all']:
        save_every = 300

    all_idx = 'all_' if config['load_embeddings_all'] else ''

    if config['pos'] is False:
        for i in range(N):
            with torch.no_grad():
                input_ids = torch.unsqueeze(dataset_ids[i], 0).to(torch.device(device))
                outs = model(input_ids, output_hidden_states=True).hidden_states[embd_layer_int].detach().cpu()
                if torch.sum(torch.isnan(outs)) != 0:
                    print('Found nan', i)
                    continue
                if not config['load_embeddings_all']:
                    outs_to_save = torch.zeros((outs.shape[0], 3, outs.shape[2]))
                    outs_to_save[:,0,:] = outs[:,0,:]
                    outs_to_save[:,1,:] = outs[:,-1,:]
                    outs_to_save[:,2,:] = torch.mean(outs, dim=1)
                else:
                    outs_to_save = outs
                model_outputs.append(outs_to_save)
                for lda_idx in range(3):
                    mixtures = lda_models[lda_idx].get_document_topics(lda_corpus[i], minimum_probability=1e-15)
                    mixtures = [tup[1] for tup in mixtures]
                    target_mixtures[lda_idx].append(mixtures)
                if i % 100 == 0:
                    print(i)
                if (i+1) % save_every == 0:
                    with open(f'data/model_outputs_{i+1}_wiki_{all_idx}{model_type}_{embd_layer}_{train_val}.pickle', 'wb') as handle:
                        pickle.dump(model_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    model_outputs = []
                if i+1 == N:
                    with open(f'data/model_outputs_{i+1}_wiki_{all_idx}{model_type}_{embd_layer}_{train_val}.pickle', 'wb') as handle:
                        pickle.dump(model_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    with open(f'data/wiki-targetmix_{all_idx}{model_type}_{embd_layer}_K{K}_{train_val}.pickle', 'wb') as handle:
                        pickle.dump(target_mixtures, handle, protocol=pickle.HIGHEST_PROTOCOL)
            torch.cuda.empty_cache()

    else:
        model_outputs_noun = []
        model_outputs_verb = []
        model_outputs_adp = []
        target_mixtures_noun = [[],[],[]]
        target_mixtures_verb = [[],[],[]]
        target_mixtures_adp = [[],[],[]]
        nlp = spacy.load("en_core_web_sm")
        for i in range(N):
            with torch.no_grad():
                input_ids = torch.unsqueeze(dataset_ids[i], 0).to(torch.device(device))
                outs = model(input_ids, output_hidden_states=True)
                outs_embd = outs.hidden_states[-1].detach().cpu()
                outs_word = torch.argmax(outs.logits.detach().cpu(), dim=2) # shape (1, num_words)
                if torch.sum(torch.isnan(outs_embd)) != 0:
                    print('Found nan', i)
                    continue
                mixtures_seeds = []
                for lda_idx in range(3):
                    mixtures = lda_models[lda_idx].get_document_topics(lda_corpus[i], minimum_probability=1e-15)
                    mixtures = [tup[1] for tup in mixtures]
                    mixtures_seeds.append(mixtures)
                noun_count = 0
                verb_count = 0
                adp_count = 0
                for w in range(-1, -50, -1):
                    word = outs_word[:,w]
                    dec = tokenizer.decode(word)
                    word_spacy = nlp(dec.strip())
                    if len(word_spacy) == 0:
                        continue
                    else:
                        word_spacy = word_spacy[-1]
                        pos = word_spacy.pos_
                        if pos == 'NOUN':
                            noun_count += 1
                            if noun_count <= 5:
                                model_outputs_noun.append(outs_embd[:,[w],:])
                                for lda_idx in range(3):
                                    target_mixtures_noun[lda_idx].append(mixtures_seeds[lda_idx])
                        if pos == 'VERB':
                            verb_count += 1
                            if verb_count <= 5:
                                model_outputs_verb.append(outs_embd[:,[w],:])
                                for lda_idx in range(3):
                                    target_mixtures_verb[lda_idx].append(mixtures_seeds[lda_idx])
                        if pos == 'ADP':
                            adp_count += 1
                            if adp_count <= 5:
                                model_outputs_adp.append(outs_embd[:,[w],:])
                                for lda_idx in range(3):
                                    target_mixtures_adp[lda_idx].append(mixtures_seeds[lda_idx])
                if i % 100 == 0:
                    print(i)
                if (i+1) % save_every == 0:
                    with open(f'data/model_outputs_noun_{i+1}_wiki_{model_type}_{train_val}.pickle', 'wb') as handle:
                        pickle.dump(model_outputs_noun, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    with open(f'data/model_outputs_verb_{i+1}_wiki_{model_type}_{train_val}.pickle', 'wb') as handle:
                        pickle.dump(model_outputs_verb, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    with open(f'data/model_outputs_adp_{i+1}_wiki_{model_type}_{train_val}.pickle', 'wb') as handle:
                        pickle.dump(model_outputs_adp, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    model_outputs_noun = []
                    model_outputs_verb = []
                    model_outputs_adp = []
                if i+1 == N:
                    with open(f'data/model_outputs_noun_{i+1}_wiki_{model_type}_{train_val}.pickle', 'wb') as handle:
                        pickle.dump(model_outputs_noun, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    with open(f'data/model_outputs_verb_{i+1}_wiki_{model_type}_{train_val}.pickle', 'wb') as handle:
                        pickle.dump(model_outputs_verb, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    with open(f'data/model_outputs_adp_{i+1}_wiki_{model_type}_{train_val}.pickle', 'wb') as handle:
                        pickle.dump(model_outputs_adp, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    with open(f'data/wiki-targetmix_noun_{model_type}_pos_{train_val}.pickle', 'wb') as handle:
                        pickle.dump(target_mixtures_noun, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    with open(f'data/wiki-targetmix_verb_{model_type}_pos_{train_val}.pickle', 'wb') as handle:
                        pickle.dump(target_mixtures_verb, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    with open(f'data/wiki-targetmix_adp_{model_type}_pos_{train_val}.pickle', 'wb') as handle:
                        pickle.dump(target_mixtures_adp, handle, protocol=pickle.HIGHEST_PROTOCOL)
            torch.cuda.empty_cache()

    if config['load_embeddings_all']:
        losses = []
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        for i in range(N):
            with torch.no_grad():
                input_ids = torch.unsqueeze(dataset_ids[i], 0).to(torch.device(device))
                logits = model(input_ids, labels=input_ids).logits
                loss = criterion(logits[0,:-1,:], input_ids[0,1:]).detach().cpu()
                losses.append(loss)
                if i % 100 == 0:
                    print(i)
                if (i+1) % save_every == 0:
                    with open(f'data/losses_{i+1}_wiki_{all_idx}{model_type}_{train_val}.pickle', 'wb') as handle:
                        pickle.dump(losses, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    losses = []
                if i+1 == N:
                    with open(f'data/losses_{i+1}_wiki_{all_idx}{model_type}_{train_val}.pickle', 'wb') as handle:
                        pickle.dump(losses, handle, protocol=pickle.HIGHEST_PROTOCOL) 
            torch.cuda.empty_cache()

    return N

def get_target_only_wiki(config, dataset_ids, lda_corpus, model, tokenizer, lda_models, K=20, train_val='train', device='cpu'):

    model_type = config['model_type']
    embd_layer = config['embd_layer']
    embd_layer_int = config['embd_layer_int']
    model_outputs = []
    target_mixtures = [[],[],[]]
    N = len(dataset_ids)
    print('length', N)
    model.eval()
    save_every = 10000
    if config['load_embeddings_all']:
        save_every = 300

    all_idx = 'all_' if config['load_embeddings_all'] else ''

    if config['pos'] is False:
        for i in range(N):
            with torch.no_grad():
                input_ids = torch.unsqueeze(dataset_ids[i], 0).to(torch.device(device))
                outs = model(input_ids, output_hidden_states=True).hidden_states[embd_layer_int].detach().cpu()
                if torch.sum(torch.isnan(outs)) != 0:
                    print('Found nan', i)
                    continue
                for lda_idx in range(3):
                    mixtures = lda_models[lda_idx].get_document_topics(lda_corpus[i], minimum_probability=1e-15)
                    mixtures = [tup[1] for tup in mixtures]
                    target_mixtures[lda_idx].append(mixtures)
                if i % 100 == 0:
                    print(i)
                if i+1 == N:
                    with open(f'data/wiki-targetmix_{all_idx}{model_type}_{embd_layer}_K{K}_{train_val}.pickle', 'wb') as handle:
                        pickle.dump(target_mixtures, handle, protocol=pickle.HIGHEST_PROTOCOL)
            torch.cuda.empty_cache()

    return N


def load_embeddings(config, N, train_val='train'):
    token = config['token']
    dataset = config['dataset']
    embd_layer = config['embd_layer']
    embd_layer_int = config['embd_layer_int']
    K = config['K']
    if dataset == '20ng':
        remove_header = config['remove_header']
        use_eos_token = config['use_eos_token']
        raw = config['raw']
        header_option = '_noheader' if remove_header else ''
        eos_option = '_eos' if use_eos_token else ''
        raw_option = '_raw' if raw else ''
        finetune = '_ft' if config['load_finetune'] != 'none' else ''
        full_len = '_full' if config['token'] == 2.5 else ''
        save_every = 200 if config['token'] == 2.5 else 1000
    elif dataset == 'wiki':
        save_every = 10000
        all_idx = 'all_' if config['load_embeddings_all'] else ''
        if config['load_embeddings_all']:
            save_every = 300
    model_type = config['model_type']

    model_outputs = []
    losses = []

    lst_tmp = [i for i in range(save_every, N+1, save_every)]
    lst_tmp.append(N)

    if dataset == '20ng':
        for i in lst_tmp:
            model_outputs.extend(pd.read_pickle(f'data/model_outputs_{i}{header_option}{eos_option}{raw_option}{finetune}{full_len}_{model_type}_{embd_layer}_{train_val}.pickle'))
            losses.extend(pd.read_pickle(f'data/losses_{i}{header_option}{eos_option}{raw_option}{finetune}{full_len}_{model_type}_{embd_layer}_K{K}_{train_val}.pickle'))

        for i, loss in enumerate(losses):
            losses[i] = losses[i].to(torch.device('cpu'))

        target_mixtures = pd.read_pickle(f'data/20ng-targetmix{header_option}{eos_option}{raw_option}{finetune}{full_len}_{model_type}_{embd_layer}_K{K}_{train_val}.pickle')
        target_mixtures = np.array(target_mixtures, dtype=np.float32)

        return target_mixtures, model_outputs, losses

    elif dataset == 'wiki':

        if not config['pos']:
            if not config['load_embeddings_all']:
                for i in lst_tmp:
                    model_output_batch = pd.read_pickle(f'data/model_outputs_{i}_wiki_{all_idx}{model_type}_{embd_layer}_{train_val}.pickle')
                    for model_output_i in model_output_batch:
                        model_outputs.append(model_output_i[:,:,:])         
            
            else:
                for i in lst_tmp:
                    # losses_batch = pd.read_pickle(f'data/losses_{i}_wiki_{all_idx}{model_type}_{train_val}.pickle')
                    # for losses_i in losses_batch:
                    #     idx = int(token*len(model_output_i[0]))
                    #     losses.append(losses_i[idx, :])
                    model_output_batch = pd.read_pickle(f'data/model_outputs_{i}_wiki_{all_idx}{model_type}_{embd_layer}_{train_val}.pickle')
                    
                    for model_output_i in model_output_batch:
                        idx = int(token*len(model_output_i[0]))
                        model_outputs.append(model_output_i[0,idx,:])
                    
                    # os.makedirs(f'data/debugcheck{i}/')
                # for i, loss in enumerate(losses):
                #     losses[i] = losses[i].to(torch.device('cpu'))

            target_mixtures = pd.read_pickle(f'data/wiki-targetmix_{all_idx}{model_type}_{embd_layer}_K{K}_{train_val}.pickle')
            target_mixtures = np.array(target_mixtures, dtype=np.float32)

            return target_mixtures, model_outputs, losses
        
        else:
            model_outputs_noun = []
            model_outputs_verb = []
            model_outputs_adp = []
            target_mixtures_noun = []
            target_mixtures_verb = []
            target_mixtures_adp = []

            for i in lst_tmp:
                # model_outputs_noun.extend(pd.read_pickle(f'data/model_outputs_noun_{i}_wiki_{model_type}_{train_val}.pickle'))
                model_outputs_verb.extend(pd.read_pickle(f'data/model_outputs_verb_{i}_wiki_{model_type}_{train_val}.pickle'))
                # model_outputs_adp.extend(pd.read_pickle(f'data/model_outputs_adp_{i}_wiki_{model_type}_{train_val}.pickle'))

            # target_mixtures_noun = pd.read_pickle(f'data/wiki-targetmix_noun_{model_type}_pos_{train_val}.pickle')
            # target_mixtures_noun = np.array(target_mixtures_noun, dtype=np.float32)
            target_mixtures_verb = pd.read_pickle(f'data/wiki-targetmix_verb_{model_type}_pos_{train_val}.pickle')
            target_mixtures_verb = np.array(target_mixtures_verb, dtype=np.float32)
            # target_mixtures_adp = pd.read_pickle(f'data/wiki-targetmix_adp_{model_type}_pos_{train_val}.pickle')
            # target_mixtures_adp = np.array(target_mixtures_adp, dtype=np.float32)

            return target_mixtures_noun, target_mixtures_verb, target_mixtures_adp, model_outputs_noun, model_outputs_verb, model_outputs_adp