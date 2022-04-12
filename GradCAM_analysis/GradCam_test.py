from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import sys
sys.path.append('/home/lab1211/BaselineRegularization')
from backbone import Conv4
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
from torchvision import transforms
import matplotlib.pyplot as plt

PILtransform = ToPILImage()

if __name__ == '__main__':
    
    model = Conv4()
    modelfile = 'checkpoints/tiered_imagenet/Conv4_baseline_aug/599.tar'
    tmp = torch.load(modelfile)
    state = tmp['state']
    state_keys = list(state.keys())
    for i, key in enumerate(state_keys):
        if "feature." in key:
            newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architect    ure feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx
            if 'module.' in newkey:
                newkey = newkey.replace("module.","")
            state[newkey] = state.pop(key)
        else:
            state.pop(key)
            
    model.load_state_dict(state)
    target_layers = model.trunk[3].trunk
    
    image_path = '/home/lab1211/Data/tiered-imagenet-tools/tiered_imagenet/val/n02099267/n0209926700000520.jpg'

    transform1 = transforms.Compose([transforms.ToTensor(), ])
    input_image = Image.open(image_path).convert('RGB')
    input_tensor = transform1(input_image).unsqueeze(0)


    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    targets = [ClassifierOutputTarget(97)]
    # targets = None

    rgb_cam = cam(input_tensor=input_tensor, targets=targets)
    input_image = np.float32(input_image) / 255
    grayscale_cam = rgb_cam[0, :]
    visualization = show_cam_on_image(input_image, grayscale_cam, use_rgb=True)
    plt.imshow(visualization)
    plt.show()
