import cv2,os,torch,json,jsonlines
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import torchvision.models as models
from transformers import AutoTokenizer,AutoModel,MPNetTokenizer
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet101
import pytorch_lightning as pl
from tqdm import tqdm
# from precompute import mean_pooling
import scipy.io as scio
import random


data_root="./data/"
coco_path="./data/coco"
f30k_path="./data/f30k"

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
def get_transform( split):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    # t_list = [transforms.Resize((224, 224))]
    # t_list = [transforms.Resize(256), transforms.CenterCrop(224)]
    if split == "train":
        t_list = [transforms.RandomResizedCrop(224),
                  transforms.RandomHorizontalFlip()]
    elif split == "val":
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]
        #t_list = [transforms.Resize((224, 224))]
    elif split== "test":
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]
        #t_list = [transforms.Resize((224, 224))]

    t_end = [transforms.ToTensor(), normalizer]

    transform = transforms.Compose(t_list + t_end)
    return transform


class coco(data.Dataset):
    def __init__(self,split='train',max_length=32,labels=None):
        super(coco, self).__init__()
        with open(os.path.join(coco_path,"splits.json"),"r") as f:
            captions=json.load(f)
        self.labels=labels
        self.split=split
        captions=captions[self.split]
        self.input_ids=[]
        self.att_masks=[]
        self.image_paths=[]
        self.tokenizer=MPNetTokenizer.from_pretrained('./tokenizer')
        for idx,item in enumerate(tqdm(captions)):
            image_path=os.path.join(coco_path,os.path.join(item["filepath"],item["filename"]))
            captions=item["sentences"][:5]
            for caption in captions:
                self.image_paths.append(image_path)
                encoded = self.tokenizer.encode_plus(caption.lower(),max_length=max_length,truncation=True,padding='max_length')
                self.input_ids.append(encoded["input_ids"])
                self.att_masks.append(encoded["attention_mask"])
        self.len=len(self.image_paths)
        if self.labels is not None:
            assert self.len==len(self.labels)
        self.transform=get_transform("val")
    def __getitem__(self, index):
        image_path=self.image_paths[index]
        image=Image.open(image_path,"r").convert("RGB")
        image=self.transform(image)
        input_id=torch.tensor(self.input_ids[index]).long()
        att_mask=torch.tensor(self.att_masks[index])
        if self.labels is not None:
            label=torch.tensor(self.labels[index]).long()
            return image,input_id,att_mask,label
        # print(image.shape,input_id.shape,att_mask.shape)
        return image,input_id,att_mask
    def __len__(self):
        return self.len


class FlickrImage(data.Dataset):
    """
    Dataset loader for Flickr30k and Flickr8k full datasets.
    Formats:
        "images":[{
            "sentids": [0, 1, 2, 3, 4],
            "imgid": 0,
            "sentences": [{
                "tokens": ["a", "sample", "example"],
                "raw": "A sample example.",
                "imgid": 0,
                "sentid": 0
            }, ... ]
            "split": "train/val/test",
            "filename:" "xxx.jpg",
        }, ... ]
    """
    def __init__(self, split, max_length=32,return_idx=False):
        self.root = f30k_path
        self.max_length=max_length
        self.split = split
        # with open("f30k_"+split+".pkl","rb") as f:
        #     self.images=cPickle.load(f)
        self.transform = get_transform("val")
        self.tokenizer=MPNetTokenizer.from_pretrained('./tokenizer')
        self.dataset = json.load(open(os.path.join(self.root,"dataset.json"), "r"))["images"]
        self.ids = []
        self.img_num = 0
        self.return_idx=return_idx
        for i, d in enumerate(tqdm(self.dataset)):
            if d["split"] == split:
                self.ids.append(i)
                self.img_num += 1

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        root = self.root
        img_id = self.ids[index]
        path = self.dataset[img_id]["filename"]
        image = Image.open(os.path.join(root, "images/"+path)).convert("RGB")
        # image=torch.tensor(self.images[index])
        caption = self.dataset[img_id]["sentences"][0]["raw"]
        encoded = self.tokenizer.encode_plus(caption.lower(),max_length=self.max_length,truncation=True,padding='max_length')
        input_ids=torch.tensor(encoded["input_ids"]).long()
        att_mask=torch.tensor(encoded["attention_mask"])
        if self.transform is not None:
            image = self.transform(image)
        if self.return_idx:
            return image,input_ids,att_mask, torch.tensor(index)
        return image,input_ids,att_mask

    def __len__(self):
        return self.img_num
        # return 2000
class FlickrDataset(data.Dataset):
    """
    Dataset loader for Flickr30k and Flickr8k full datasets.
    Formats:
        "images":[{
            "sentids": [0, 1, 2, 3, 4],
            "imgid": 0,
            "sentences": [{
                "tokens": ["a", "sample", "example"],
                "raw": "A sample example.",
                "imgid": 0,
                "sentid": 0
            }, ... ]
            "split": "train/val/test",
            "filename:" "xxx.jpg",
        }, ... ]
    """
    def __init__(self, split, max_length=32,return_idx=False):
        self.root = f30k_path
        self.max_length=max_length
        self.split = split
        self.transform = get_transform(split)
        self.tokenizer=MPNetTokenizer.from_pretrained('./tokenizer')
        self.dataset = json.load(open(os.path.join(self.root,"dataset.json"), "r"))["images"]
        self.ids = []
        self.img_num = 0
        self.return_idx=return_idx
        for i, d in enumerate(tqdm(self.dataset)):
            if d["split"] == split:
                self.ids += [(i, x, self.img_num) for x in range(len(d["sentences"]))]
                self.img_num += 1

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        root = self.root
        ann_id = self.ids[index]
        img_id = ann_id[0]
        img_cls = ann_id[2]
        caption = self.dataset[img_id]["sentences"][ann_id[1]]["raw"]
        encoded = self.tokenizer.encode_plus(caption.lower(),max_length=self.max_length,truncation=True,padding='max_length')
        input_ids=torch.tensor(encoded["input_ids"]).long()
        att_mask=torch.tensor(encoded["attention_mask"])
        path = self.dataset[img_id]["filename"]
        image = Image.open(os.path.join(root, "images/"+path)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.return_idx:
            return image, input_ids,att_mask,torch.tensor(index)
        return image, input_ids,att_mask

    def __len__(self):
        return len(self.ids)
        # return 2000

class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.file_path = path
        self.dataset = None
        with h5py.File(self.file_path, 'r') as file:
            self.dataset_len = len(file["dataset"])

    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')["dataset"]
        return self.dataset[index]

    def __len__(self):
        return self.dataset_len
import h5py

coco_pre_path=".data/coco/coco_precompute.hdf5"

class precompute(data.Dataset):
    def __init__(self,split='train',dataset='coco',labels=None,return_idx=False):
        super(precompute, self).__init__()
        self.file_path = data_root+dataset+"_precompute.hdf5"
        self.dataset_name = dataset
        self.dataset = None
        self.labels=labels
        with h5py.File(self.file_path, 'r') as file:
            if dataset=="unknown":
                self.dataset_len = len(file["captions"])
            else:
                self.dataset_len = len(file[split]["captions"])
        self.split=split
        self.return_idx=return_idx
    def __getitem__(self, index):
        if self.dataset is None:
            if self.dataset_name=="unknown":
                self.dataset =  h5py.File(self.file_path, 'r')
            else:
                self.dataset = h5py.File(self.file_path, 'r')[self.split]
        image=torch.tensor(self.dataset["images"][index])
        caption=torch.tensor(self.dataset["captions"][index])
        if self.labels is not None:
            label=torch.tensor(self.labels[index]).long()
            return image,caption,label
        elif self.return_idx:
            return image,caption, torch.tensor(index)
        return image,caption

    def __len__(self):
        return self.dataset_len
import  re

   
class precompute_image(data.Dataset):
    def __init__(self,split='train',dataset='coco',max_length=32,return_idx=False):
        super(precompute_image, self).__init__()
        self.per=5
        if dataset=='cub':
            self.per=10
        self.file_path = data_root+dataset+"_precompute.hdf5"
        self.dataset = None
        self.return_idx=return_idx
        with h5py.File(self.file_path, 'r') as file:
            self.dataset_len = len(file[split]["images"])//self.per
        self.split=split
    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')[self.split]
        image=torch.tensor(self.dataset["images"][index*self.per])
        caption=torch.tensor(self.dataset["captions"][index*self.per])
        if self.return_idx:
            return image,caption, torch.tensor(index)
        return image,caption
        # return image

    def __len__(self):
        return self.dataset_len

class precompute_caption(data.Dataset):
    def __init__(self,split='train',dataset='coco',max_length=32):
        super(precompute_caption, self).__init__()
        self.per=5
        if dataset=='cub':
            self.per=10
        self.file_path = data_root+dataset+"_precompute.hdf5"
        self.dataset = None
        with h5py.File(self.file_path, 'r') as file:
            self.dataset_len = len(file[split]["captions"])//self.per
        self.split=split
    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')[self.split]
        caption=torch.tensor(self.dataset["captions"][index*self.per])
        # caption=torch.tensor(self.dataset["captions"][index])
        return caption

    def __len__(self):
        return self.dataset_len

class precompute_captiontotal(data.Dataset):
    def __init__(self,split='train'):
        super(precompute_captiontotal, self).__init__()
        self.file_path = [coco_pre_path]
        self.datasets = None
        self.lens=[]
        for path in self.file_path:
            with h5py.File(path, 'r') as file:
                self.lens.append(len(file[split]["captions"]))
        self.dataset_len=self.lens[0]+self.lens[1]
        self.split=split
    def __getitem__(self, index):
        if self.datasets is None:
            self.datasets = [h5py.File(self.file_path[0], 'r')[self.split],h5py.File(self.file_path[1], 'r')[self.split]]
        if index<self.lens[0]:
            caption=torch.tensor(self.datasets[0]["captions"][index])
        else:
            index=index-self.lens[0]
            caption=torch.tensor(self.datasets[1]["captions"][index])
        return caption

    def __len__(self):
        return self.dataset_len
class simulate_subset(data.Dataset):
    def __init__(self,image_idxs,caption_idxs,split='train'):
        super(simulate_subset, self).__init__()
        self.split=split
        self.image_idxs=image_idxs
        self.caption_idxs=caption_idxs
        assert len(image_idxs)==len(caption_idxs)
        self.image_dataset=None
        # self.caption_datasets=None
        self.file_path = [coco_pre_path]
        # self.lens=[]
        # for path in self.file_path:
        #     with h5py.File(path, 'r') as file:
        #         self.lens.append(len(file[split]["captions"]))
        self.dataset_len=len(image_idxs)
    def __getitem__(self, index):
        if self.image_dataset is None:
            self.image_dataset = h5py.File(self.file_path[1], 'r')[self.split]["images"]
        # if self.caption_datasets is None:
        #     self.caption_datasets = [h5py.File(self.file_path[0], 'r')[self.split]["captions"],
        #                              h5py.File(self.file_path[1], 'r')[self.split]["captions"]]
        image_idx=self.image_idxs[index]
        # caption_idx=self.caption_idxs[index]
        image=torch.tensor(self.image_dataset[image_idx])
        cap_idx=self.caption_idxs[index]
        # if index<self.lens[0]:
        #     caption=torch.tensor(self.caption_datasets[0][caption_idx])
        # else:
        #     caption_idx=caption_idx-self.lens[0]
        #     caption=torch.tensor(self.caption_datasets[1][caption_idx])
        return image,cap_idx

    def __len__(self):
        return self.dataset_len

import _pickle as cPickle
class WrapLoader(pl.LightningDataModule):
    def __init__(self, args,dataset):
        super().__init__()
        self.dataset=dataset
        self.batch_sz=args.batch_sz
        self.num_workers=args.num_workers
        self.train_source_data=None
        self.train_target_data=None
        self.val_source_data=None
        self.test_target_data=None
        self.stage=None
    def setup(self, stage=None):
        if stage == 'fit':
            if self.dataset=="precompute":
                self.train_source_data=precompute("train","coco")
                self.val_source_data=precompute("val","coco")
                self.val_target_data=precompute("val","cub")
            
            elif self.dataset=="coco-f30k-only-source":
                # self.train_source_data=coco("train")
                # self.val_source_data=coco("val")
                self.train_source_data=precompute("train","coco")
                self.val_source_data=precompute("val","coco")
            elif self.dataset=="coco-f30k":
                labels=cPickle.load(open("./data/labels.pkl","rb"))
                self.train_source_data=coco("train",labels=labels)
                self.val_source_data=coco("val")
                self.val_target_data=FlickrDataset("val")
            elif self.dataset=="coco-f30k-precompute":
                labels=cPickle.load(open("./data/labels.pkl","rb"))
                self.train_source_data=precompute("train","coco",labels=labels)
                self.val_source_data=precompute("val","coco")
                self.val_target_data=precompute("val","f30k")
            elif self.dataset=="coco-f30k-all":
                labels = cPickle.load(open("./labels.pkl", "rb"))
                # self.train_source_data=coco("train",labels=labels)
                # self.val_source_data=coco("val")
                # self.val_target_data=FlickrDataset("val")
                self.train_source_data = precompute("train", "coco", labels=labels)
                self.val_source_data = precompute("val", "coco")
                self.val_target_data=precompute("val","f30k")
            elif self.dataset=="f30k":
                # self.train_source_data=FlickrDataset("train")
                # self.val_source_data=FlickrDataset("val")
                self.train_source_data=precompute("train","f30k")
                self.val_source_data=precompute("val","f30k")
            elif self.dataset=="f30k-sample":
                self.train_source_data=precompute_image("train","f30k")
                self.val_source_data=precompute("val","f30k")
            else:
                self.train_source_data=precompute("train","coco")
                self.val_source_data=precompute("val","coco")
        elif stage == 'test':
            if self.dataset in ["coco-f30k-only-source", "f30k", "coco-f30k", "coco-f30k-all"]:
                self.test_target_data = precompute("test", "f30k")
            elif self.dataset in ["coco-f30k-precompute", "f30k-sample"]:
                self.test_target_data=precompute("test","f30k")
            elif self.dataset=="coco":
                self.test_target_data=coco("val")
           


    def train_dataloader(self):
        if self.dataset in["coco-f30k","f30k-sample","coco-f30k-precompute","tgif-msr"]:
            if self.dataset=="tgif-msr":
                indices=random.sample(range(0,len(self.train_source_data)),5000)
            elif self.dataset=="msr-sample":
                indices=random.sample(range(0,len(self.train_source_data)),400*12)
            else:
                indices=random.sample(range(0,len(self.train_source_data)),6000) #从训练数据中随机选择 6000 个索引
            sampler=data.SubsetRandomSampler(indices)
            source_loader = DataLoader(
                self.train_source_data,
                batch_size=self.batch_sz,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=False
            )
        else:
            source_loader = DataLoader(
                self.train_source_data,
                batch_size=self.batch_sz,
                # sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=False
            )

        # else:
        #     source_loader = DataLoader(
        #         self.train_source_data,
        #         batch_size=self.batch_sz,
        #         shuffle=True,
        #         num_workers=self.num_workers,
        #         pin_memory=False
        #     )
        #     if self.dataset!="baseline" and self.dataset!="cub" and self.dataset!= "coco-f30k-only-source" and self.dataset!= "f30k":
        #         target_loader = DataLoader(
        #             self.train_target_data,
        #             batch_size=self.batch_sz,
        #             shuffle=True,
        #             num_workers=self.num_workers,
        #             pin_memory=False
        #         )
        #         return [source_loader,target_loader]
        return source_loader

    def val_dataloader(self):
        source_loader=DataLoader(
            self.val_source_data,
            batch_size=self.batch_sz,
            shuffle=False,
            num_workers=1, # 1
            pin_memory=False
        )
        if self.dataset in["precompute","coco-f30k" ,"coco-f30k-all","coco-f30k-precompute","tgif-msr","tgif-msr-warm"] :

            target_loader=DataLoader(
                self.val_target_data,
                batch_size=self.batch_sz,
                shuffle=False,
                num_workers=1, # 1
                pin_memory=False
            )
            loaders = [source_loader,target_loader]
            return loaders
        return source_loader
    def test_dataloader(self):
        loader=DataLoader(
            self.test_target_data,
            batch_size=self.batch_sz,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False
        )
        return loader