from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import sys
sys.path.append('/home/lab1211/BaselineRegularization')
from backbone import Conv4, Conv4Drop
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

PILtransform = ToPILImage()

def load_modelfile(model, modelfile):
    tmp = torch.load(modelfile)
    state = tmp['state']
    state_keys = list(state.keys())
    for i, key in enumerate(state_keys):
        if "feature." in key:
            newkey = key.replace("feature.","")
            if 'module.' in newkey:
                newkey = newkey.replace("module.","")
            state[newkey] = state.pop(key)
        else:
            state.pop(key)
            
    model.load_state_dict(state)
    return model

def read_imgs(img_path):
    transform1 = transforms.Compose([transforms.ToTensor(), ])
    input_image = Image.open(img_path).convert('RGB')
    input_tensor = transform1(input_image).unsqueeze(0)
    input_image = np.float32(input_image) / 255
    return input_tensor, input_image
    
if __name__ == '__main__':
    modellist = [Conv4(), Conv4Drop()]
    ckp_path = 'checkpoints/tiered_imagenet/'
    # methodlist = ['Conv4_baseline_aug', 'Conv4Drop_baseline_aug', 'Conv4_baselinecutmix_aug', 'Conv4_baselinelabelsmoothing_aug']
    methodlist = ['Conv4_baselinelabelsmoothing_aug']
    tar_file = '599.tar'
    modelfilelist = [os.path.join(ckp_path, item, tar_file) for item in methodlist]

    IMAGE_ROOT = '/home/lab1211/Data/tiered-imagenet-tools/tiered_imagenet/test/'
    class_file = 'n01494475'
    IMAGE_ROOT = os.path.join(IMAGE_ROOT, class_file)
    img_path_list = os.listdir(IMAGE_ROOT)
    img_path_list.sort(key = lambda x: int(x[-8:-4]))
    for index, item in enumerate(methodlist):
        if 'Drop' in item:
            model = load_modelfile(modellist[1], modelfilelist[index])
        else:
            model = load_modelfile(modellist[0], modelfilelist[index])

        target_layers = model.trunk[3].trunk
        for i, img_path in enumerate(img_path_list):
            img_path = os.path.join(IMAGE_ROOT, img_path)
            input_tensor, input_image = read_imgs(img_path)
            # cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
            cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)
            targets = [ClassifierOutputTarget(160)]
            rgb_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = rgb_cam[0, :]
            visualization = show_cam_on_image(input_image, grayscale_cam, use_rgb=True)
            save_path_root = 'GradCAM_analysis/save_imgs/tiered_imagenet'
            save_path = os.path.join(save_path_root, item, str(i + 1) + '.png')
            mpimg.imsave(save_path, visualization)
