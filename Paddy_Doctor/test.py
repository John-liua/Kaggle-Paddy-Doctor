# -*- coding: utf-8 -*

# -------------------------------------------------------------------------------
# Author: LiuNing
# Contact: 2742229056@qq.com
# Software: PyCharm
# File: test.py
# Time: 6/27/19 3:00 PM
# Description: 
# -------------------------------------------------------------------------------
import pandas as pd

from core.config import *
from core.model import *
from dataload.dataloader import *
from tqdm import tqdm
import json

init_environment()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
multi_gpus = False

diseases = ['normal', 'downy_mildew', 'tungro', 'brown_spot', 'blast', 'bacterial_leaf_streak',
            'bacterial_leaf_blight', 'hispa', 'dead_heart', 'bacterial_panicle_blight']
submission_output = "./result/sample_submission.csv"

def main():
    # # testing different classes: 'danyi', 'jiandan', 'fuza'
    # test_names = ['danyi', 'jiandan', 'fuza']

    # for test_name in test_names:

    test_iter = load_unlabeled_data(
        root='./database',
        test_paths=['./datalist/test.txt'],
        resize_size=RESIZE_SIZE,
        input_size=INPUT_SIZE,
        batch_size=4,
        num_workers=2
    )

    net = agriculture_model1(classes=10)
    net.load_state_dict(torch.load('./model/model.ckpt')['net_state_dict'])

    net = net.cuda()
    if multi_gpus:
        net = nn.DataParallel(net)

    net.eval()
    img_ids, labels = [], []
    for data in tqdm(test_iter, desc='Test'):
        with torch.no_grad():
            inputs, image_names = data
            inputs = inputs.cuda()
            b_size = inputs.size(0)

            logits = net.features(inputs)[0]
            for i in range(b_size):
                img_ids.append(image_names[i].split('/')[-1])
                labels.append(diseases[int(torch.argmax(logits[i]).data.cpu().numpy())])
    print(labels)

    submission = pd.DataFrame({
        'image_id': img_ids,
        'label': labels,
    })

    submission.to_csv(submission_output, index=False, header=True)

    print('finish')


if __name__ == '__main__':
    main()
