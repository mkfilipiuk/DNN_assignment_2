import os
import json
import numpy as np
import random
import shutil
import torch

from PIL import Image

colours_list = [
     (116, 17, 36),
     (152, 43,150),
     (106,141, 34),
     ( 69, 69, 69),
     (  2,  1,  3),
     (127, 63,126),
     (222, 52,211),
     (  2,  1,140),
     ( 93,117,119),
     (180,228,182),
     (213,202, 43),
     ( 79,  2, 80),
     (188,151,155),
     (  9,  5, 91),
     (106, 75, 13),
     (215, 20, 53),
     (110,134, 62),
     (  8, 68, 98),
     (244,171,170),
     (171, 43, 74),
     (104, 96,155),
     ( 72,130,177),
     (242, 35,231),
     (147,149,149),
     ( 35, 25, 34),
     (155,247,151),
     ( 85, 68, 99),
     ( 71, 81, 43),
     (195, 64,182),
     (146,133, 92)
]

colours_dict = dict(zip(colours_list, range(len(colours_list))))
id_to_colour_dict = dict(zip(range(len(colours_list)), colours_list))

def load_config():
    with open('config.json', 'r') as f:
        config = json.load(f)
    return config

# taken from https://stackoverflow.com/questions/7762948/how-to-convert-an-rgb-image-to-numpy-array
def load_image(infilename):
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

# Taken from https://stackoverflow.com/questions/33196130/replacing-rgb-values-in-numpy-array-by-integer-is-extremely-slow
def img_array_to_single_val(arr, color_codes):
    result = np.ndarray(shape=arr.shape[:2], dtype=int)
    result[:,:] = -1
    for rgb, idx in color_codes.items():
        result[(arr==rgb).all(2)] = idx
    return result

def preprocess_data(config, validation_set_size = 0.1, test_set_size = 0):
    files = os.listdir(config["RAW_DATA_PATH"])
    number_of_pictures = len([name for name in files if name.endswith(".png")])
    random.shuffle(files)
    inputs, outputs = [], []
    for n, filename in enumerate(files):
        if filename.endswith(".png"): 
            whole_img = load_image(os.path.join(config["RAW_DATA_PATH"],filename))
            img_input = whole_img[:,:256,:]
            img_coded_output = img_array_to_single_val(whole_img[:,256:,:], colours_dict)
            inputs.append(img_input)
            outputs.append(img_coded_output)
    inputs_np, outputs_np = np.stack(inputs), np.stack(outputs)
    
    np.save(os.path.join(config["PREPROCESSED_DATA_PATH"], "input"),inputs_np)
    np.save(os.path.join(config["PREPROCESSED_DATA_PATH"], "output"),outputs_np)
                            
def load_dataset(config, validation_set_size = 0.1, test_set_size = 0.1, force_preprocessing = False):
    if not os.path.exists(os.path.join(os.getcwd(),config["PREPROCESSED_DATA_PATH"])):
        os.mkdir(config["PREPROCESSED_DATA_PATH"])
        for folder in ["train", "validation", "test"]:
            os.mkdir(os.path.join(config["PREPROCESSED_DATA_PATH"], folder))                        
        preprocess_data(config)
    elif force_preprocessing:
        shutil.rmtree(config["PREPROCESSED_DATA_PATH"])
        os.mkdir(config["PREPROCESSED_DATA_PATH"])
        for folder in ["train", "validation", "test"]:
            os.mkdir(os.path.join(config["PREPROCESSED_DATA_PATH"], folder))
        preprocess_data(config)
        
    input_np = np.load(os.path.join(config["PREPROCESSED_DATA_PATH"], "input.npy"))
    output_np = np.load(os.path.join(config["PREPROCESSED_DATA_PATH"], "output.npy"))
    mask = np.arange(input_np.shape[0])
    np.random.shuffle(mask)
   
    number_of_pictures = mask.shape[0]
    test_size = int(number_of_pictures*test_set_size)
    validation_size = int(number_of_pictures*validation_set_size)
   
    return ((torch.tensor((input_np[mask])[test_size+validation_size:]).permute(0,3,1,2).contiguous(), 
            torch.tensor((output_np[mask])[test_size+validation_size:])),
            (torch.tensor((input_np[mask])[test_size:test_size+validation_size]).permute(0,3,1,2).contiguous(), torch.tensor((output_np[mask])[test_size:test_size+validation_size])),
            (torch.tensor((input_np[mask])[:test_size]).permute(0,3,1,2).contiguous(), torch.tensor((output_np[mask])[:test_size])))
           
def build_set_loader(set_tuple, training=True, batch_size = 8):
    set_input, set_output = set_tuple
    if training:
        set_input_flipped = torch.flip(set_input.clone(),[3])
        set_output_flipped = torch.flip(set_output.clone(),[2])
        dataset = torch.utils.data.TensorDataset(torch.cat((set_input,set_input_flipped)).float(),
                                                 torch.cat((set_output,set_output_flipped)))
    else:
        dataset = torch.utils.data.TensorDataset(set_input.float(),
                                                 set_output)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)