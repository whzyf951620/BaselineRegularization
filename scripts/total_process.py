import os

method_list = ['baseline', 'baseline++', 'baselinecutmix', 'baseline++cutmix', \
        'baseline_labelsmoothing', 'baseline++labelsmoothing', 'baselinemixup', \
        'baselinecutmixlabelsmoothing']

for method in method_list:
    save_path = os.path.join('./record', method + '.txt')
    os.system('python train.py --model Conv4 --method ' + method + ' --dataset miniImagenet --train_aug')
    os.system('python save_features.py --model Conv4 --method ' + method + ' --dataset miniImagenet --train_aug')
    os.system('python test.py --model Conv4 --method ' + method + ' --dataset miniImagenet --train_aug > ' + save_path + ' 2>&1')
