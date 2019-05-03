import os
import json
import numpy as np
import random
import shutil

from PIL import Image
from sklearn.preprocessing import OneHotEncoder

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

def preprocess_data(config, validation_set_size = 0.1, test_set_size = 0.1):
    ohe = OneHotEncoder(categories='auto', sparse=False, dtype=np.int32)
    ohe.fit(img_array_to_single_val(load_image(os.path.join(config["RAW_DATA_PATH"],"0001.png"))[:,256:,:], colours_dict).reshape([-1,1]))
    number_of_pictures = len([name for name in os.listdir(config["RAW_DATA_PATH"]) if name.endswith(".png")])
    files = os.listdir(config["RAW_DATA_PATH"])
    random.shuffle(files)
    for n, filename in enumerate(files):
        if filename.endswith(".png"): 
            img = load_image(os.path.join(config["RAW_DATA_PATH"],filename))
            img_coded = img_array_to_single_val(img[:,256:,:], colours_dict)
            #img_ohe = ohe.transform(img_coded.reshape([-1,1])).reshape(256,256,30)
            if n < number_of_pictures*validation_set_size:
                folder = "validation"
            elif n < number_of_pictures*(validation_set_size+test_set_size):
                folder = "test"
            else:
                folder = "train"
                
            # data augumentation - flips
            np.save(os.path.join(config["PREPROCESSED_DATA_PATH"], folder, filename + "_output"), img_coded)
            np.save(os.path.join(config["PREPROCESSED_DATA_PATH"], folder, filename + "_input"), img[:,:256,:])
            np.save(os.path.join(config["PREPROCESSED_DATA_PATH"], folder, filename + "_flipped_output"), np.fliplr(img_coded))
            np.save(os.path.join(config["PREPROCESSED_DATA_PATH"], folder, filename + "_flipped_input"), np.fliplr(img[:,:256,:]))

def aux_loading(config, folder, split_flips):
    inputs, outputs = [], []
    if split_flips:
        inputs_flipped, outputs_flipped = [], []
    for f in sorted(os.listdir(os.path.join(config["PREPROCESSED_DATA_PATH"], folder))):
        loaded_picture = np.load(os.path.join(config["PREPROCESSED_DATA_PATH"], folder, f))
        if split_flips:
            if "output" in f:
                if "flipped" in f:
                    outputs_flipped.append(loaded_picture)
                else:
                    outputs.append(loaded_picture)
            else:
                if "flipped" in f:
                    inputs_flipped.append(loaded_picture)
                else:
                    inputs.append(loaded_picture)
        else:
            if "output" in f:
                outputs.append(loaded_picture)
            else:
                inputs.append(loaded_picture)
    if split_flips:
        return ((np.stack(inputs, axis=0), np.stack(inputs_flipped, axis=0)), (np.stack(outputs_flipped, axis=0), np.stack(outputs_flipped, axis=0)))
    else:
        return (np.stack(inputs, axis=0), np.stack(outputs, axis=0))
            
 
            
            
def load_train(config):
    return aux_loading(config, "train", split_flips=False)

def load_validation(config):
    return aux_loading(config, "validation", split_flips=True)

def load_test(config):
    return aux_loading(config, "test", split_flips=True)

            
                            
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
    
    train_input, train_output = load_train(config)
    ((validation_input,validation_input_flipped), (validation_output,validation_output_flipped)) = load_validation(config)
    ((test_input,test_input_flipped), (test_output,test_output_flipped)) = load_test(config)
    
    return ((train_input, train_output), 
           ((validation_input,validation_input_flipped), (validation_output,validation_output_flipped)),
           ((test_input,test_input_flipped), (test_output,test_output_flipped)))
