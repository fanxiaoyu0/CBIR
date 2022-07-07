import PIL
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import pickle
import os

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

def get_image_name_list(file_path):
    image_name_list=[]
    image_name_file=open(file_path,'r')
    for line in image_name_file.readlines():
        line=line.strip()
        if line=='':
            continue
        image_path=line.split(' ')[0]
        image_class=image_path.split('/')[0]
        image_name=image_path.split('/')[1]
        image_name_list.append((image_class,image_name))
    return image_name_list

def get_all_image_bin_vector(bins):
    image_name_to_vector_dict={}
    image_name_list=get_image_name_list('../data/AllImages.txt')
    for (image_class,image_name) in image_name_list:
        image=Image.open('../data/image/'+image_class+'/'+image_name)
        w, h = image.size  
        colors = image.getcolors(w*h)
        bin_tensor=get_bin_tensor(bins,colors)
        bin_vector=bin_tensor.flatten()
        bin_vector=bin_vector/(w*h)
        if image_class not in image_name_to_vector_dict.keys():
            image_name_to_vector_dict[image_class]={}
        image_name_to_vector_dict[image_class][image_name]=bin_vector
    pickle.dump(image_name_to_vector_dict,open('../data/intermediate/image_name_to_vector_dict_'+str(bins)+'_bins.pkl','wb'))

def get_L1_distance(P,Q):
    return np.sum(np.abs(P-Q))

def get_L2_distance(P,Q):
    return np.sqrt(np.sum(np.square(P-Q)))

def get_HI_similarity(P,Q):
    return np.sum(np.min(np.stack([P,Q]),axis=0))/np.sum(Q)

def get_Bh_distance(P,Q):
    return np.sqrt(1-np.sum(np.sqrt(P*Q)))

def get_similar_image_list(bins,Q):
    L1_distance_list=[]
    L2_distance_list=[]
    HI_similarity_list=[]
    Bh_distance_list=[]
    image_name_to_vector_dict=pickle.load(open('../data/intermediate/image_name_to_vector_dict_'+str(bins)+'_bins.pkl','rb'))
    for class_name in image_name_to_vector_dict.keys():
        for image_name in image_name_to_vector_dict[class_name].keys():
            P=image_name_to_vector_dict[class_name][image_name]
            L1_distance_list.append((class_name,image_name,get_L1_distance(P=P,Q=Q)))
            L2_distance_list.append((class_name,image_name,get_L2_distance(P=P,Q=Q)))
            HI_similarity_list.append((class_name,image_name,get_HI_similarity(P=P,Q=Q)))
            Bh_distance_list.append((class_name,image_name,get_Bh_distance(P=P,Q=Q)))
    return L2_distance_list,HI_similarity_list,Bh_distance_list

def process_query_list(bins):
    if os.path.exists('../result/'+str(bins)+'bins/L1/res_overall.txt'):
        os.remove('../result/'+str(bins)+'bins/L1/res_overall.txt')
    if os.path.exists('../result/'+str(bins)+'bins/L2/res_overall.txt'):
        os.remove('../result/'+str(bins)+'bins/L2/res_overall.txt')
    if os.path.exists('../result/'+str(bins)+'bins/HI/res_overall.txt'):
        os.remove('../result/'+str(bins)+'bins/HI/res_overall.txt')
    if os.path.exists('../result/'+str(bins)+'bins/Bh/res_overall.txt'):
        os.remove('../result/'+str(bins)+'bins/Bh/res_overall.txt')
    L1_precision_list=[]
    L2_precision_list=[]
    HI_precision_list=[]
    Bh_precision_list=[]
    query_image_name_list=get_image_name_list('../data/QueryImages.txt')
    for (query_image_class,query_image_name) in query_image_name_list:
        image=Image.open('../data/image/'+query_image_class+'/'+query_image_name)
        w, h = image.size  
        colors = image.getcolors(w*h)
        bin_tensor=get_bin_tensor(bins,colors)
        bin_vector=bin_tensor.flatten()
        bin_vector=bin_vector/(w*h)
        L2_distance_list,HI_similarity_list,Bh_distance_list=get_similar_image_list(bins=bins,Q=bin_vector)
        L2_distance_list.sort(key=lambda x:x[2])
        HI_similarity_list.sort(key=lambda x:x[2],reverse=True)
        Bh_distance_list.sort(key=lambda x:x[2])
        with open('../result/'+str(bins)+'bins/L2/res_'+query_image_class+'_'+query_image_name[:-4]+'.txt','w') as f:
            for (class_name,image_name,distance) in L2_distance_list[:30]:
                f.write(class_name+'/'+image_name[:-4]+' '+str(distance)+'\n')
        with open('../result/'+str(bins)+'bins/HI/res_'+query_image_class+'_'+query_image_name[:-4]+'.txt','w') as f:
            for (class_name,image_name,distance) in HI_similarity_list[:30]:
                f.write(class_name+'/'+image_name[:-4]+' '+str(distance)+'\n')
        with open('../result/'+str(bins)+'bins/Bh/res_'+query_image_class+'_'+query_image_name[:-4]+'.txt','w') as f:
            for (class_name,image_name,distance) in Bh_distance_list[:30]:
                f.write(class_name+'/'+image_name[:-4]+' '+str(distance)+'\n')
        L2_precision=np.sum([1 for (class_name,image_name,distance) in L2_distance_list[:30] if class_name==query_image_class])/30
        HI_precision=np.sum([1 for (class_name,image_name,distance) in HI_similarity_list[:30] if class_name==query_image_class])/30
        Bh_precision=np.sum([1 for (class_name,image_name,distance) in Bh_distance_list[:30] if class_name==query_image_class])/30
        L2_precision_list.append(L2_precision)
        HI_precision_list.append(HI_precision)
        Bh_precision_list.append(Bh_precision)
        with open('../result/'+str(bins)+'bins/L2/res_overall.txt','a') as f:
            f.write(query_image_class+'/'+query_image_name[:-4]+' '+str(L2_precision)+'\n')
        with open('../result/'+str(bins)+'bins/HI/res_overall.txt','a') as f:
            f.write(query_image_class+'/'+query_image_name[:-4]+' '+str(HI_precision)+'\n')
        with open('../result/'+str(bins)+'bins/Bh/res_overall.txt','a') as f:
            f.write(query_image_class+'/'+query_image_name[:-4]+' '+str(Bh_precision)+'\n')
    with open('../result/'+str(bins)+'bins/L2/res_overall.txt','a') as f:
        f.write(str(np.mean(L2_precision_list))+'\n')
    with open('../result/'+str(bins)+'bins/HI/res_overall.txt','a') as f:
        f.write(str(np.mean(HI_precision_list))+'\n')
    with open('../result/'+str(bins)+'bins/Bh/res_overall.txt','a') as f:
        f.write(str(np.mean(Bh_precision_list))+'\n')    

if __name__ == '__main__':
    # get_all_image_bin_vector(16)
    # get_all_image_bin_vector(128)
    process_query_list(16)
    process_query_list(128)
    print("All is well!")
