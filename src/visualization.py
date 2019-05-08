import numpy as np
import matplotlib.pyplot as plt
import torch

def ids_to_rgb_image_pytorch(arr, color_codes):
    arr = arr.argmax(dim=0)
    result = torch.zeros(256,256,3)
    for idx, rgb  in color_codes.items():
        result[(arr==idx)] = torch.Tensor(list(rgb))
    return result.permute([2,0,1])

def ids_to_rgb_image(arr, id_to_colour_dict):
    arr = arr.argmax(dim=0).cpu().long().view(-1)
    result = torch.zeros((256,256,3), dtype=torch.uint8)
    result[:,:,:] = -1
    result = result.view(-1,3)
    for idx, rgb  in id_to_colour_dict.items():
        mask = (arr==idx)
        result[mask] = torch.Tensor(list(rgb)).type(torch.uint8)
    return result.view(256,256,3)

def show_image(img, title=None):
    plt.imshow(img)
    plt.axis('off')
    plt.title(title)
    plt.show()