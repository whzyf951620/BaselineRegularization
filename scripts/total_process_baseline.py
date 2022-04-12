import os

# method_list = ['baseline', 'baselinecutmix', 'baseline_labelsmoothing', 'baselinemixup', 'baselinecutmixlabelsmoothing']
method_list = ['baselinecutmixlabelsmoothing']

for method in method_list:
    save_path = os.path.join('./record', 'Conv4' + method + '.txt')
    os.system('python train.py --model Conv4 --method ' + method + ' --dataset miniImagenet --train_aug')
    os.system('python save_features.py --model Conv4 --method ' + method + ' --dataset miniImagenet --train_aug')
    os.system('python test.py --model Conv4 --method ' + method + ' --dataset miniImagenet --train_aug > ' + save_path + ' 2>&1')
