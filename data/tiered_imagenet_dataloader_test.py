import numpy as np
from tiered_imagenet_dataloader import TieredImageNetDataLoader
dataloader = TieredImageNetDataLoader(shot_num=5, way_num=5, episode_test_sample_num=15)

dataloader.generate_data_list(phase='train')
dataloader.generate_data_list(phase='val')
dataloader.generate_data_list(phase='test')

dataloader.load_list(phase='all')

total_train_step = 1000
for idx in range(total_train_step):
    episode_train_img, episode_train_label, episode_test_img, episode_test_label = \
        dataloader.get_batch(phase='train', idx=idx)
        
    print(np.nonzero(episode_test_label)[1])
