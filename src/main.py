import PIL
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import pickle

def get_bin_tensor(bins,color_count_list):
    if bins==16:
        red_bins=2
        green_bins=4
        blue_bins=2
    elif bins==128:
        red_bins=4
        green_bins=8
        blue_bins=4
    bin_tensor=np.zeros((red_bins,green_bins,blue_bins),dtype=np.float32)
    for color_count in color_count_list:
        red_index=color_count[1][0]//(256//red_bins)
        green_index=color_count[1][1]//(256//green_bins)
        blue_index=color_count[1][2]//(256//blue_bins)
        bin_tensor[red_index][green_index][blue_index]+=color_count[0]
    return bin_tensor

def get_all_image_bin_vector(bins):
    image_name_to_vector_dict={}
    image_file=open('../data/AllImages.txt','r')
    for line in tqdm(image_file.readlines()):
        line=line.strip()
        if line=='':
            continue
        image_path=line.split(' ')[0]
        image_class=image_path.split('/')[0]
        image_name=image_path.split('/')[1]
        image=Image.open('../data/image/'+image_class+'/'+image_name)
        w, h = image.size  
        colors = image.getcolors(w*h)
        bin_tensor=get_bin_tensor(bins,colors)
        # print(bin_tensor)
        bin_vector=bin_tensor.flatten()
        # print(bin_vector)
        bin_vector=bin_vector/(w*h)
        # print(bin_vector)
        # print(np.sum(bin_vector))
        if image_class not in image_name_to_vector_dict.keys():
            image_name_to_vector_dict[image_class]={}
        image_name_to_vector_dict[image_class][image_name]=bin_vector
        # print(image_name_to_vector_dict)
        # break
    pickle.dump(image_name_to_vector_dict,open('../data/intermediate/image_name_to_vector_dict.pkl','wb'))

def get_L2_distance(P,Q):
    return np.sqrt(np.sum(np.square(P-Q)))

def get_HI_distance(P,Q):
    return np.sum(np.min(np.concatenate((P,Q),axis=1),axis=1))/np.sum(Q)

def get_Bhattacharyya_distance(P,Q):
    return np.sqrt(1-np.sum(np.sqrt(P*Q)))

def get_similar_image_list(query):


if __name__ == '__main__':
    get_all_image_bin_vector(16)

    print("All is well!")