# coding: UTF-8
import time
import torch
import numpy as np
from importlib import import_module
from utils import build_dataset, build_iterator, get_time_dif
from train_eval import train, init_network
import argparse

if __name__ == '__main__':
    dataset = 'tmpdataset'
    embedding = 'embedding_SougouNews.npz'
    model_name = 'TextRCNN'
    word = True

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    init_network(model)
    train(config, model, train_iter, dev_iter, test_iter)
