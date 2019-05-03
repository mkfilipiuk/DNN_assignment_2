import os
import json
import numpy as np

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

def preprocess_data(config):
    ohe = OneHotEncoder(categories='auto', sparse=False, dtype=np.int32)
    ohe.fit(img_array_to_single_val(load_image(os.path.join(config["RAW_DATA_PATH"],"0001.png"))[:,256:,:], colours_dict).reshape([-1,1]))
    for filename in os.listdir(config["RAW_DATA_PATH"]):
        if filename.endswith(".png"): 
            img = load_image(os.path.join(config["RAW_DATA_PATH"],filename))[:,256:,:]
            img_coded = img_array_to_single_val(img, colours_dict)
            ohe.transform(img_coded.reshape([-1,1])).reshape(256,256,30)
            np.save(os.path.join(config["PREPROCESSED_DATA_PATH"],filename), img_coded)
        
    

def load_dataset(config, validation_set_size = 0.1, test_set_size = 0.1, force_preprocessing = False):
    if not os.path.exists(os.path.join(os.getcwd(),config["PREPROCESSED_DATA_PATH"])):
        os.mkdir(config["PREPROCESSED_DATA_PATH"])
        preprocess_data(config)
    elif force_preprocessing:
        os.rmdir(config["PREPROCESSED_DATA_PATH"])
        preprocess_data(config)
    
    