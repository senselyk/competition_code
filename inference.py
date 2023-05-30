# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif


dataset = 'tmpdataset'
model_name = 'bert'
embedding = 'embedding_SougouNews.npz'
if model_name in ['bert','bert_CNN','ERNIE']:
    from utils_bert import build_dataset, build_iterator, get_time_dif
    from train_eval_bert import test
else:
    from utils import build_dataset, build_iterator, get_time_dif
    from train_eval import test

x = import_module('models.' + model_name)
try:
    config = x.Config(dataset, embedding)
except:
    config = x.Config(dataset)

start_time = time.time()
print("Loading data...")
try:
    vocab, train_data, dev_data, test_data = build_dataset(config, True)
except:
    train_data, dev_data, test_data = build_dataset(config)
test_iter = build_iterator(test_data, config)
time_dif = get_time_dif(start_time)
print("Time usage:", time_dif)
try:
    config.n_vocab = len(vocab)
except:
    pass
model = x.Model(config).to(config.device)

print(model.parameters)
test(config, model, test_iter)
