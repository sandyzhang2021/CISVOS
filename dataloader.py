
import numpy as np
import torch
from torch.utils.data import Dataset
import os.path
from glob import glob
import imageio
from misc import imutils
import myutils
from PIL import Image
from misc.pyutils import to_one_hot
import cv2
import torchvision.transforms as TF
from dataset import transforms as mytrans
IMG_FOLDER_NAME = "JPEGImages"
ANNOT_FOLDER_NAME = "Annotations"
IGNORE = 255
from collections import defaultdict
# from skimage.io import imread_collection
# import skimage.io as io


CAT_LIST = ['bear', 'bmx-bumps', 'boat', 'boxing-fisheye',
        'breakdance-flare', 'bus', 'car-turn', 'cat-girl', 'classic-car',
        'color-run', 'crossing', 'dance-jump', 'dancing',
        'disc-jockey', 'dog-agility', 'dog-gooses',
        'dogs-scale', 'drift-turn', 'drone','elephant','flamingo','hike','hockey',
        'horsejump-low', 'kid-football','kite-walk','koala','lady-running','lindy-hop',
        'longboard','lucia','mallard-fly','mallard-water','miami-surf','motocross-bumps',
        'motorbike','night-race','paragliding','planes-water','rallye','rhino','rollerblade',
        'schoolgirls','scooter-board','scooter-gray','sheep','skate-park','snowboard',
        'soccerball','stroller','stunt','surf','swing','tennis','tractor-sand','train',
        'tuk-tuk','upside-down','varanus-cage','walking']
# cls_labels_dict = np.load(r'/media/a208/新加卷/zz/code/AFB/cls_labels2.npy', allow_pickle=True).item()
# print(cls_labels_dict)

N_CAT = len(CAT_LIST)

CAT_NAME_TO_NUM = dict(zip(CAT_LIST,range(len(CAT_LIST))))
# print("CAT_NAME_TO_NUM :",CAT_NAME_TO_NUM)

def decode_long_filename(int_filename):
    # s = str(int(int_filename))
    s = str(int_filename)
    return s[:10] + '_' + s[10:]

def load_image_from_label(img_name):
   
    # img_dir = os.path.join(img_name)
    # elem_list = sorted(glob(os.path.join(img_dir, '*.jpg')))
    # idx_list = list(range(len(img_list)))
    # elem_list = minidom.parse(os.path.join(root, ANNOT_FOLDER_NAME, decode_long_filename(img_name) + '.xml')).getElementsByTagName('name')

    cls_label = np.zeros((N_CAT), np.float32)

    # for elem in elem_list:
    # if img_dir in CAT_LIST:
    #     cat_num = CAT_NAME_TO_NUM[img_dir]
    #     cls_label[cat_num] = 1.0
    # print("cls_label:",cls_label)
    return cls_label

def load_image_label_list_from_xml(img_name_list, root):

    return [load_image_from_label(img_name, root) for img_name in img_name_list]



def load_image_label_list(img_name, root):
    img_dir = os.path.join(root, 'JPEGImages', '480p', img_name)
    img_list = sorted(glob(os.path.join(img_dir, '*.jpg')))
    idx_list = list(range(len(img_list)))
    
    return [load_image_label_list(img_name, root) for img_name in idx_list]
        
        

    return [load_image_label_from_xml(img_name, voc12_root) for img_name in img_name_list]

def load_image_label_list_from_npy(root,name_list):
    sequences = defaultdict(dict)
    # cls_labels_dict = defaultdict(dict)
    mask_names = os.path.join(root, 'Annotations', '480p')
    for seq in name_list:
            masks = np.sort(glob(os.path.join(mask_names, seq, '*.png'))).tolist() 
            mks = [img[-9:] for img in masks]
            # print("mks :",mks)     
            if len(masks) == 0 :
                raise FileNotFoundError(f'Images for sequence {seq} not found.')
            sequences[seq] = mks
            return np.array(sequences[seq])



def get_img_path(img_name, root):
    img_dir = os.path.join(root, 'JPEGImages', '480p', img_name)
    
    img_name_list = sorted(glob(os.path.join(img_dir, '*.jpg')))

    return img_name_list


def load_img_name_list(root,name):
    # img_name_list = np.loadtxt(dataset_path, dtype=np.int32)

    # return img_name_list
    imagesets_path=os.path.join(root,'ImageSets',name)
    img_path = os.path.join(root,'JPEGImages', '480p')
    with open(imagesets_path) as f:
        tmp=f.readlines()
    sequences_names = [x.strip() for x in tmp] #
    sequences = defaultdict(dict)
    for seq in sequences_names:
            images = np.sort(glob(os.path.join(img_path, seq, '*.jpg'))).tolist()
            imgs = [img[-9:] for img in images]
            # print("imgs.size:",imgs)       
            if len(images) == 0 :
                raise FileNotFoundError(f'Images for sequence {seq} not found.')
            sequences[seq] = imgs
    
            # print("sequences f{[seq}=",sequences[seq])
    

    return sequences

def img_name_list(root,name):
    
    img_path = os.path.join(root,'JPEGImages', '480p',name)

    # images_names = np.sort(glob(os.path.join(img_path, '*.jpg')))
    images_names  = sorted(glob(os.path.join(img_path, '*.jpg')))
    # print('images_names',images_names) 
    imgs = list([img[-9:-4] for img in images_names])
    # print("imgs.size:",imgs)
          
    

    return imgs,images_names

def load_img_name_list2(root,image_name_list):
    masksets_path=os.path.join(root,'ImageSets',image_name_list)
    mask_path = os.path.join(root,'Annotations', '480p')
    with open(masksets_path) as f:
        tmp=f.readlines()
    sequences_names = [x.strip() for x in tmp] #
    sequences = defaultdict(dict)
    for seq in sequences_names:
            masks = np.sort(glob(os.path.join(mask_path, seq, '*.png'))).tolist()
            mks = [img[-9:] for img in masks]
            # print("imgs.size:",mks)       
            if len(masks) == 0 :
                raise FileNotFoundError(f'Images for sequence {seq} not found.')
            sequences[seq] = mks
    

    return sequences
    # img_name_list = np.loadtxt(dataset_path, dtype=np.int32)

    return img_name_list





class TorchvisionNormalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img

class GetAffinityLabelFromIndices():

    def __init__(self, indices_from, indices_to):

        self.indices_from = indices_from
        self.indices_to = indices_to

    def __call__(self, segm_map):

        segm_map_flat = np.reshape(segm_map, -1)

        segm_label_from = np.expand_dims(segm_map_flat[self.indices_from], axis=0)
        segm_label_to = segm_map_flat[self.indices_to]

        valid_label = np.logical_and(np.less(segm_label_from, 21), np.less(segm_label_to, 21))

        equal_label = np.equal(segm_label_from, segm_label_to)

        pos_affinity_label = np.logical_and(equal_label, valid_label)

        bg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(segm_label_from, 0)).astype(np.float32)
        fg_pos_affinity_label = np.logical_and(pos_affinity_label, np.greater(segm_label_from, 0)).astype(np.float32)

        neg_affinity_label = np.logical_and(np.logical_not(equal_label), valid_label).astype(np.float32)

        return torch.from_numpy(bg_pos_affinity_label), torch.from_numpy(fg_pos_affinity_label), \
               torch.from_numpy(neg_affinity_label)

class VOC12ImageDataset2(Dataset):

    def __init__(self, root,img_name_list_path,
                 resize_long=None, rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_size=None, crop_method=None, to_torch=True):
        self.root = root
        # self.img_name_list = load_img_name_list(root,img_name_list_path)
        self.img_name_list = list(load_img_name_list(root,img_name_list_path).keys())
        self.resize_long = resize_long
        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method
        self.to_torch = to_torch

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        # name_str = decode_int_filename(name)
        img_dir = os.path.join(self.root, 'JPEGImages', '480p', name)
        imgs=[]
        for filename in os.listdir(img_dir):            
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img_path = os.path.join(img_dir, filename)
                # img = Image.open(img_path)
                img =imageio.imread(img_path)
                img = np.asarray(img)
        # img = np.asarray(imageio.imread(get_img_path(name_str, self.voc12_root)))

                if self.resize_long:
                    img = imutils.random_resize_long(img, self.resize_long[0], self.resize_long[1])

                if self.rescale:
                    img = imutils.random_scale(img, scale_range=self.rescale, order=3)

                if self.img_normal:
                    img = self.img_normal(img)

                if self.hor_flip:
                    img = imutils.random_lr_flip(img)

                if self.crop_size:
                    if self.crop_method == "random":
                        img = imutils.random_crop(img, self.crop_size, 0)
                    else:
                        img = imutils.top_left_crop(img, self.crop_size, 0)

                if self.to_torch:
                    img = imutils.HWC_to_CHW(img)
            imgs.append(img)
        return {'name': name, 'img': imgs}

class VOC12ImageDataset(Dataset):
    def __init__(self, root,img_name_list_path,
                 resize_long=None, rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_size=None, crop_method=None, to_torch=True):
        self.root = root
        self.name = list(load_img_name_list(root,img_name_list_path).keys())
        self.img_name_list = load_img_name_list(root,img_name_list_path)
        self.mask_name_list = load_img_name_list2(root,img_name_list_path)
        self.resize_long = resize_long
        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method
        self.index = list(self.img_name_list.keys()) 
        self.to_torch = to_torch
        # self.to_tensor = TF.ToTensor()
        
    def __len__(self):
        lenth = len(self.img_name_list)
        return lenth

    def __getitem__(self, idx):
        # name = self.name[idx] 
        # name = idx 
        return idx 
        # return {'name': name}
    

class VOC12ClassificationDataset(VOC12ImageDataset):

    def __init__(self, root,img_name_list_path,
                 resize_long=None, rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_size=None, crop_method=None):
        super().__init__(root,img_name_list_path,
                 resize_long, rescale, img_normal, hor_flip,
                 crop_size, crop_method)
        # self.root = root
        self.name = list(load_img_name_list(root,img_name_list_path).keys())
        self.label_list = load_image_label_list_from_npy(root,self.mask_name_list)
     
                    
    def __getitem__(self, idx):        
        out = super().__getitem__(self.name[idx])
        img_dir = os.path.join(self.root, 'JPEGImages', '480p', self.name[idx])
        img_list = sorted(glob(os.path.join(img_dir, '*.jpg')))
        cls_label = np.zeros((N_CAT), np.float32)
        cat_name = out
        if cat_name in CAT_LIST:
            cat_num = CAT_NAME_TO_NUM[cat_name]
            cls_label[cat_num] = 1.0
        # print("cls_label",cls_label)
        imgs=[]
        for filename in os.listdir(img_dir):            
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img_path = os.path.join(img_dir, filename)
                    
                # img = Image.open(img_path)
                img =imageio.imread(img_path)
                img = np.asarray(img)
                if self.rescale:
                    img= imutils.random_scale(img, scale_range=self.rescale, order=(3, 0))
                # img = imutils.pil_rescale(img, 0.25, 0)
                if self.img_normal:
                    img = self.img_normal(img)
        
                if self.hor_flip:
                    img = imutils.random_lr_flip(img)

                if self.crop_method == "random":
                    img = imutils.random_crop(img, self.crop_size, (0, 255))
                else:
                    img = imutils.top_left_crop(img, self.crop_size, 0)
    
                img = imutils.HWC_to_CHW(img)
                imgs.append(img)
                # imgs[i] = torch.Tensor(img) 

        
        
        # imgs = np.array(imgs)
        # imgs = torch.Tensor(imgs)
        return {'name':out,'img':imgs,'label':torch.from_numpy(cls_label)}
    
        
    
class VOC12ClassificationDatasetMSF(VOC12ClassificationDataset):

    def __init__(self, root,img_name_list_path,
                 img_normal=TorchvisionNormalize(),
                 scales=(1.0,)):
        self.scales = scales
        self.name = list(load_img_name_list(root,img_name_list_path).keys())
        
        super().__init__(root,img_name_list_path,img_normal=img_normal)
        self.scales = scales

    def __getitem__(self, idx):
        name = self.name[idx]
        img_dir = os.path.join(self.root, 'JPEGImages', '480p', self.name[idx]) 
        ms_img_list = []
        # sizes = []
        for filename in os.listdir(img_dir):
            if filename.endswith(".jpg"):
                img_path = os.path.join(img_dir, filename)
                # img = Image.open(img_path)
                img =imageio.imread(img_path)
                img = np.asarray(img)
                # img = imutils.pil_rescale2(img, 0.2, 0)
                for s in self.scales:
                    if s == 1:
                        s_img = img
                    else:
                        s_img = imutils.pil_rescale(img, s, order=3)#对原始图像进行resize
                    s_img = self.img_normal(s_img)
                    size = (s_img.shape[0],s_img.shape[1])
                    s_img = imutils.HWC_to_CHW(s_img)
                
            ms_img_list.append(np.stack([s_img, np.flip(s_img, -1)], axis=0))
        cls_label = np.zeros((N_CAT), np.float32)
        cat_name = self.name[idx]
        if cat_name in CAT_LIST:
            cat_num = CAT_NAME_TO_NUM[cat_name]
            cls_label[cat_num] = 1.0   
        return {"name": name, "img": ms_img_list, "size": size,"label": torch.from_numpy(cls_label)}

class VOC12ClassificationDatasetMSF2(VOC12ClassificationDataset):

    def __init__(self, root,img_name_list_path,
                 img_normal=TorchvisionNormalize(),
                 scales=(1.0,)):
        self.scales = scales
        self.name = list(load_img_name_list(root,img_name_list_path).keys())
        
        super().__init__(root,img_name_list_path,img_normal=img_normal)
        self.scales = scales

    def __getitem__(self, idx):
        name = self.name[idx]
        img_dir = os.path.join(self.root, 'JPEGImages', '480p', self.name[idx]) 
        ms_img_list = []
        # sizes = []
        for filename in os.listdir(img_dir):
            if filename.endswith(".jpg"):
                img_path = os.path.join(img_dir, filename)
                # img = Image.open(img_path)
                img =imageio.imread(img_path)
                # img =imageio.imread(img_path)
                img = np.asarray(img)
                # img = imutils.pil_rescale2(img, 0.2, 0)
                for s in self.scales:
                    if s == 1:
                        s_img = img
                    else:
                        s_img = imutils.pil_rescale(img, s, order=3)#对原始图像进行resize
                    s_img = self.img_normal(s_img)
                    size = (s_img.shape[0],s_img.shape[1])
                    s_img = imutils.HWC_to_CHW(s_img)
                
            ms_img_list.append(np.stack([s_img, np.flip(s_img, -1)], axis=0))
        # if len(self.scales) == 1:
        #     ms_img_list = ms_img_list[0]
        cls_label = np.zeros((N_CAT), np.float32)
        cat_name = self.name[idx]
        if cat_name in CAT_LIST:
            cat_num = CAT_NAME_TO_NUM[cat_name]
            cls_label[cat_num] = 1.0   
        return {"name": name, "img": ms_img_list, "size": size,"label": torch.from_numpy(cls_label)}


class VOC12SegmentationDataset(Dataset):

    def __init__(self, root,img_name_list_path, label_dir, crop_size,
                 rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_method = 'random'):
        self.root = root
        self.img_name_list = list(load_img_name_list(root,img_name_list_path).keys())
        # self.name = list(img_name_list)
        self.label_dir = label_dir
        
        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method
        # self.to_tensor = TF.ToTensor()

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        img_dir = os.path.join(self.root, 'JPEGImages', '480p', name)
        mask_dir = os.path.join(self.root, 'Annotations', '480p', name)     
        
        
        for filename in os.listdir(img_dir):
            img_list = []                        
            if filename.endswith(".jpg"):
                img_path = os.path.join(img_dir, filename)  
                img = Image.open(img_path)
                # img = imageio.imread(img_path)
                img = np.asarray(img)
            img_list.append(img)               
        for filename in os.listdir(mask_dir): 
            png_list = []           
            if filename.endswith(".png"):
                png_path = os.path.join(mask_dir, filename)   
                png = Image.open(png_path).convert("RGB")
                # png = imageio.imread(png_path)
                png = np.asarray(png)
            png_list.append(png)
        # idx_list = list(range(len(img_list)))
        lenth = len(img_list)
        for i in range(lenth):
                
            if self.rescale:
                img_list[i], png_list[i] = imutils.random_scale((img_list[i], png_list[i]), scale_range=self.rescale, order=(3, 0))

            if self.img_normal:
                img_list[i] = self.img_normal(img_list[i])
                png_list[i] = self.img_normal(png_list[i])

            if self.hor_flip:
                img_list[i], png_list[i] = imutils.random_lr_flip((img_list[i], png_list[i]))

            if self.crop_method == "random":
                img_list[i], png_list[i] = imutils.random_crop((img_list[i], png_list[i]), self.crop_size, (0, 255))
            else:
                img_list[i] = imutils.top_left_crop(img_list[i], self.crop_size, 0)
                png_list[i] = imutils.top_left_crop(png_list[i], self.crop_size, 255)
            img_list[i] = imutils.HWC_to_CHW(img_list[i])
            png_list[i] = imutils.HWC_to_CHW(png_list[i])
        # img_list = torch.Tensor(img_list)
        # png_list = torch.Tensor(png_list)
            
        out = {'name': name, 'img': img_list, 'label': png_list}
        
        return out

class VOC12AffinityDataset(VOC12SegmentationDataset):
    def __init__(self,root,img_name_list_path, label_dir, crop_size, 
                 indices_from, indices_to,
                 rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False, crop_method=None):
        super().__init__(root,img_name_list_path, label_dir, crop_size, rescale, img_normal, hor_flip, crop_method=crop_method)
        self.extract_aff_lab_func = GetAffinityLabelFromIndices(indices_from, indices_to)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        png_label= out['label']
        reduced_label = []
        aff_bg_pos_label = []
        aff_fg_pos_label = []
        aff_neg_label = []
        # for i in range(len(out['label'])):
        # reduced_label = [imutils.pil_rescale(img, 0.25, 0) for img in png_label]
        for i in png_label:
            # print("i:",i[0])
            reduced = imutils.pil_rescale(i[0], 0.25, 0)
            # reduced_label = imutils.pil_rescale(out['label'], 0.25, 0)
            a, b, c = self.extract_aff_lab_func(reduced)
            
            # out['aff_bg_pos_label'], out['aff_fg_pos_label'], out['aff_neg_label'] = self.extract_aff_lab_func(reduced)
            reduced_label.append(reduced)
            aff_bg_pos_label.append(a)
            aff_fg_pos_label.append(b)
            aff_neg_label.append(c)
        out['aff_bg_pos_label'] = aff_bg_pos_label
        out['aff_fg_pos_label'] = aff_fg_pos_label
        out['aff_neg_label'] = aff_neg_label
        # out = {'aff_bg_pos_label':aff_bg_pos_label,'aff_fg_pos_label':aff_fg_pos_label,
        #        'aff_neg_label':aff_neg_label}
        return out