import pickle
import h5py,re,torch
from tqdm import tqdm
dataset_name="unknown_msr"

with open(path,"rb") as file:
    dataset=pickle.load(file)
hfile=h5py.File(hdf5_file,"w")
if dataset_name=="unknown_msr":
    data=dataset
    length=len(data)
    ids=hfile.create_dataset("ids",(length,))
    images=hfile.create_dataset("images",(length,4096))
    captions=hfile.create_dataset("captions",(length,768))
    idx=0
    for item in tqdm(data):
        id=item["id"]
        id=re.findall("[0-9]+",id)
        id=int(id[0])
        id=torch.tensor(id)
        image=item["image"]
        caption=item["caption"]
        ids[idx]=id
        images[idx,:]=image
        captions[idx,:]=caption
        idx+=1
    hfile.close()
else:
    splits=["train","val","test"]
    for split in splits:
        data=dataset[split]
        length=len(data)
        group=hfile.create_group(split)
        ids=group.create_dataset("ids",(length,))
        images=group.create_dataset("images",(length,4096))
        captions=group.create_dataset("captions",(length,768))
        idx=0
        for item in tqdm(data):
            id=item["id"]
            id=re.findall("[0-9]+",id)
            id=int(id[0])
            id=torch.tensor(id)
            image=item["image"]
            caption=item["caption"]
            ids[idx]=id
            images[idx,:]=image
            captions[idx,:]=caption
            idx+=1
    hfile.close()
