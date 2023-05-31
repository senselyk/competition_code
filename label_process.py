import os
import pandas as pd
import ipdb
special_case = {'A12':'SR71','Tu142':'Tu95'}
label_dict = ['A10', 'AG600', 'B1', 'B2', 'B52', 'Be200', 'C130', 'C17', 'C5', 'E2', 'EF2000', 'F117', 'F14', 'F15', 'F16', 'F18', 'F22', 'F35', 'F4', 'J20', 'JAS39', 'Mig31','MQ9', 'Mirage2000', 'RQ4', 'Rafale', 'SR71','Su57', 'Tu160', 'Tu95', 'U2', 'US2', 'V22', 'XB70', 'YF23']
def change_annotations(old_anno):
    width,height,xmin,xmax,ymin,ymax = old_anno
    new_width = (xmax-xmin)/width
    new_height = (ymax-ymin)/height
    x_center,y_center = (xmax+xmin)/width/2,(ymax+ymin)/height/2
    return x_center,y_center,new_width,new_height
    
def read_csv(csv_file,txt_path):
    #读取原始标注并进行coco格式的改造，存到目标路径的txt
    f = pd.read_csv(csv_file)
    result = []

    for i in range(len(f['filename'])):
        label = f['class'][i]
        annotation = [f['width'][i],f['height'][i],f['xmin'][i],f['xmax'][i],f['ymin'][i],f['ymax'][i]]
        if label in special_case.keys():
            label = special_case[label]
        if label not in label_dict:
            print(label)
            break
        label_idx = label_dict.index(label)
        x_center,y_center,new_width,new_height = change_annotations(annotation)
        new_anno = [str(label_idx), str(x_center),str(y_center),str(new_width),str(new_height)]
        result.append(new_anno)
    save_path = os.path.join(txt_path,csv_file.split('/')[-1].replace('.csv','.txt'))
    with open(save_path,'w') as g:
        for item in result:
            g.write(' '.join(item))
            g.write('\n')

if __name__ == '__main__':
    annolist = open('./anno_list.txt','r').readlines()
    txt_path = '../data/labels/train/'
    for item in annolist:
        read_csv(item.strip(),txt_path)
        # print('Done {}'.format(item))
