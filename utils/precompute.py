import argparse
import _pickle as cPickle
import pickle

import h5py
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
import torch,os
from torch.utils.data import DataLoader
from transformers import MPNetModel,MPNetTokenizer
from torchvision.models import resnet101
import torch.nn as nn
from dataset import coco,cub,FlickrImage,FlickrDataset,unknown
os.environ["CUDA_VISIBLE_DEVICES"]='3'
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
resnet=resnet101(pretrained=True)
resnet.fc=nn.Sequential()
extractor=resnet.cuda()
extractor.eval()
sentence_model=MPNetModel.from_pretrained('./models').cuda()
sentence_model.eval()
def f30k():
    dataset="f30k"
    splits=["train","val","test"]
    for split in splits:
        data=FlickrImage(split)
        dataloader=DataLoader(
            data,
            batch_size=120,
            shuffle=False,
            num_workers=10,
            pin_memory=False
        )
        images=[]
        for i_batch, sample_batched in enumerate(tqdm(dataloader)):
            image,_,_=sample_batched
            with torch.no_grad():
                v_embed=extractor(image.cuda()).cpu().numpy()
                images.append(v_embed)
        images=np.concatenate(images,0)
        with open(dataset+"_"+split+".pkl","wb") as f:
            cPickle.dump(images,f)
def pre():
    dataset_list=["coco","f30k"]
    for dataset_name in dataset_list:
        per=5
        path=dataset_name+"_precompute.hdf5"
        file=h5py.File(path,"w")
        splits=["train","val","test"]
        for split in splits:
            if dataset_name=="coco":
                data=coco(split)
            elif dataset_name=="f30k":
                data=FlickrDataset(split)
            else:
                data=cub(split)
            dataloader=DataLoader(
                data,
                batch_size=per,#cub 每张图片有10个caption,COCO 5
                shuffle=False,
                num_workers=10,
                pin_memory=False
            )
            length=len(data)
            group=file.create_group(split)
            images=group.create_dataset("images",(length,2048))
            captions=group.create_dataset("captions",(length,768))

            for i_batch, sample_batched in enumerate(tqdm(dataloader)):
                image,input_id,att_mask=sample_batched
                input_id=input_id.cuda()
                att_mask=att_mask.cuda()
                with torch.no_grad():
                    v_embed=extractor(image.cuda())
                    v_embed=v_embed.cpu()
                    s_embed=sentence_model(input_ids=input_id,attention_mask=att_mask)
                s_embed=mean_pooling(s_embed, att_mask).cpu()
                images[i_batch*per:(i_batch+1)*per,:]=v_embed
                captions[i_batch*per:(i_batch+1)*per,:]=s_embed

        file.close()

def pre_captions():
    path="./data/collaborative-experts/data/activity-net/structured-symlinks/raw-captions-train-val_1.pkl"
    path="./data/collaborative-experts/data/MSVD/structured-symlinks/raw-captions.pkl"
    f=open(path,"rb")
    data=pickle.load(f)
    tokenizer=MPNetTokenizer.from_pretrained('./tokenizer')
    path_pre="./data/collaborative-experts/data/MSVD/structured-symlinks/raw-captions_pre.pkl"
    dicts={}
    for key, captions in enumerate(tqdm(data)):
        pre_caps=[]
        for caption in captions:
            caption=" ".join(caption)
            encoded = tokenizer.encode_plus(caption.lower(),max_length=32,truncation=True,padding='max_length')
            input_id=torch.tensor(encoded["input_ids"]).long()
            att_mask=torch.tensor(encoded["attention_mask"])
            input_id=input_id.unsqueeze(0).cuda()
            att_mask=att_mask.unsqueeze(0).cuda()
            with torch.no_grad():
                s_embed=sentence_model(input_ids=input_id,attention_mask=att_mask)
            s_embed=mean_pooling(s_embed, att_mask).cpu()
            pre_caps.append(s_embed)
        dicts[key]=pre_caps
    pickle.dump(dicts,open(path_pre,"wb"))


# f30k()
pre()
