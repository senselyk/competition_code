import pandas as pd
import random
import ipdb
from tqdm import trange

def read_csv(csv_file):
    #读取原始文件，返回按label划分的列表
    f = pd.read_csv(csv_file,sep='\t')
    list_all = {}
    list_sub = {'童书':[],'工业技术':[],'大中专教材教辅':[]}
    for i in trange(len(f['title'])):
        title = f['title'][i]
        label = f['label'][i]
        if label in list_sub.keys():
            list_sub[label].append(title)
        if label not in list_all.keys():
            list_all[label] = [title]
        else:
            list_all[label].append(title)
    print('finish process csv file...')
    return list_all,list_sub

def split_train_val(total_list):
    #处理划分train,val，并转存为txt
    train,test = [],[]
    split_ratio = 10
    for k in total_list.keys():
        lenth = len(total_list[k])
        print('start process label {}'.format(k))
        random.shuffle(total_list[k])
        tmp_train,tmp_test = [],[]
        for i in trange(lenth):
            if i < lenth//split_ratio:
                tmp_test.append(total_list[k][i])
            else:
                tmp_train.append(total_list[k][i])
        train.append(tmp_train)
        test.append(tmp_test)
    return train,test

def write_file(file_path,item_list):
    with open(file_path,'w') as f:
        for i in range(len(item_list)):
            for item in item_list[i]:
                f.write('\t'.join([item,str(i)]))
                f.write('\n')
    
if __name__ == '__main__':
    list_all,list_sub = read_csv('dev.csv')
    train,test = split_train_val(list_sub)
    write_file('/mnt/lustre/liyukun/competition/textclassification/Chinese-Text-Classification-Pytorch/tmpdataset/data/train.txt',train)
    write_file('/mnt/lustre/liyukun/competition/textclassification/Chinese-Text-Classification-Pytorch/tmpdataset/data/test.txt',test)