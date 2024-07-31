"""
clustering sentences to K clusters
prototypes are the centers of the clusters
assign a label to each sample
save the assigned labels and prototype corresponding to each label
"""
import _pickle as cPickle
from sklearn.cluster import MiniBatchKMeans
from transformers import AutoModel
import torch
# import matplotlib.pyplot as plt
import numpy as np
from dataset import precompute, precompute_image, pre_video_image
from torch.utils.data import DataLoader
from baseline import l2norm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '7'
dataset = precompute("train", "coco")
sentence_model = AutoModel.from_pretrained('./models')
sentence_model=sentence_model.cuda()

#Load data
def cluster_vide(path):
    data=cPickle.load(open(path,"rb"))
    filename=path.split(".")[0]
    # print(filename)
    captions=[]
    ids=[]
    for item in data["train"]:
        name=item["id"]
        caption=item["caption"]
        captions.append(caption)
        ids.append(name)
    captions=torch.cat(captions,0)
    # print(captions.shape)
    if "tgif" in filename:
        assert len(set(ids))==78799
    else:
        assert len(set(ids))==6513
    kmeans = MiniBatchKMeans(n_clusters=512, random_state=0)
    loader = DataLoader(
        captions,
        batch_size=1000,
        # sampler=sampler,
        num_workers=8,
        pin_memory=False,
        shuffle=False,
        drop_last=False
    )
    for idx,batch in enumerate(loader):
        caption=batch
        caption=caption.cuda()
        with torch.no_grad():
            embed=l2norm(caption)
        embed=embed.cpu().numpy()
        kmeans.partial_fit(embed)
    labels=[]
    for idx,batch in enumerate(loader):
        caption=batch
        caption=caption.cuda()
        with torch.no_grad():
            embed=l2norm(caption)
        embed=embed.cpu().numpy()
        label=kmeans.predict(embed)
        labels.extend(label)

    cPickle.dump(kmeans, open(filename+"_kmeans.pkl", "wb"))
    cPickle.dump(labels,open(filename+"_labels.pkl", "wb"))
def clustering():
    # dataset=coco("train")
    loader = DataLoader(
        dataset,
        batch_size=1000,
        # sampler=sampler,
        num_workers=8,
        pin_memory=False,
        shuffle=False
    )

    kmeans = MiniBatchKMeans(n_clusters=512, random_state=0)
    for idx,batch in enumerate(loader):
        image,caption=batch
        image,caption=image.cuda(),caption.cuda()
        with torch.no_grad():
            embed=l2norm(caption)
        embed=embed.cpu().numpy()
        kmeans.partial_fit(embed)
    cPickle.dump(kmeans, open("kmeans.pkl", "wb"))
def cluster_img(dataset_name,split):
    dataset=precompute_image(dataset=dataset_name,split=split)
    loader = DataLoader(
        dataset,
        batch_size=2000,
        # sampler=sampler,
        num_workers=8,
        pin_memory=False,
        shuffle=False
    )

    kmeans = MiniBatchKMeans(n_clusters=1024, random_state=0)
    for idx,batch in enumerate(loader):
        image,caption=batch
        image,caption=image.cuda(),caption.cuda()
        with torch.no_grad():
            embed=l2norm(image)
        embed=embed.cpu().numpy()
        kmeans.partial_fit(embed)
    cPickle.dump(kmeans, open(dataset_name+"_"+split+"_kmeans.pkl", "wb"))
def cluster_vimg(dataset_name,split):
    dataset=pre_video_image(dataset=dataset_name,split=split)
    loader = DataLoader(
        dataset,
        batch_size=2000,
        # sampler=sampler,
        num_workers=8,
        pin_memory=False,
        shuffle=False
    )

    kmeans = MiniBatchKMeans(n_clusters=1024, random_state=0)
    for idx,batch in enumerate(loader):
        _,image,caption=batch
        image,caption=image.cuda(),caption.cuda()
        with torch.no_grad():
            embed=l2norm(image)
        embed=embed.cpu().numpy()
        kmeans.partial_fit(embed)
    cPickle.dump(kmeans, open(dataset_name+"_"+split+"_kmeans.pkl", "wb"))
def testing(kmeans):

    loader = DataLoader(
        dataset,
        batch_size=1000,
        # sampler=sampler,
        num_workers=8,
        pin_memory=False,
        shuffle=False
    )
    embeds=[]
    labels=[]
    for idx,batch in enumerate(loader):
        # image,input_id,att_mask=batch
        # input_id,att_mask=input_id.cuda(),att_mask.cuda()
        # with torch.no_grad():
        #     embed=sentence_model(input_id,att_mask)
        #     embed=mean_pooling(embed, att_mask)
        #     embed=l2norm(embed)
        image,caption=batch
        embed=l2norm(caption.cuda())
        embed=embed.cpu().numpy()
        embeds.append(embed)
        label=kmeans.predict(embed)
        labels.append(label)
    labels=np.concatenate(labels,axis=None)
    old_labels=kmeans.labels_
    # labels==old_labels
    cPickle.dump(labels, open("labels.pkl", "wb"))
    cPickle.dump(old_labels, open("old_labels.pkl", "wb"))
    # embeds=np.concatenate(embeds,axis=0)
    # centers=kmeans.cluster_centers_
    # f=open("log.txt","w")
    # for i in range(512):
    #     means=np.mean(embeds[labels==i],axis=0)
    #     f.write(str(means[:20])+"\t"+str(centers[i][:20]))
    #     print(means==centers[i])

# clustering()
# cluster_img("f30k","train")
cluster_vimg("msr","train")

