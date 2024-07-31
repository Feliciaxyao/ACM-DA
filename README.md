# Active Cross-modal Domain Adaptation
## Introduction

This is the Pytorch implementation of Curiosity-Driven Active Adaptation Network (CD-A2N) for ACM-DA.


## Prerequisites
### 1. Environments
* Require packages are listed in 'requirements.txt'. You can install by running:

```
conda create -n cda2n python=3.8
pip install -r requirements.txt
```

### 2. Data Preparation: MSCOCO to Flickr30k (Image-Text Retrieval)

#### (A) Dataset Download

* Please download raw MSCOCO images, you can get access of the dataset from: https://cocodataset.org/#home
* Please download raw Flickr30k images, you can get access of the dataset from:
* Please download corresponding json blobs (i.e., dataset_coco.json, dataset_f30k.json), you can get access of the dataset from: https://cs.stanford.edu/people/karpathy/deepimagesent/
* Create coco/splits.json using:
```
import json 
with open("dataset_coco.json","r") as f: 
 captions=json.load(f) 
images=captions['images'] 
#contain 123287 images 
split={"train":[],"val":[],"test":[],"restval":[]} 
for item in images: 
 filepath=item['filepath'] 
 name=item['filename'] 
 split_name=item["split"] 
 sentences=[t["raw"] for t in item["sentences"]] 
 split[split_name].append({"filepath":filepath,"filename":name,"sentences":sentences}) 
with open("coco/splits.json","w") as f: 
 json.dump(split,f)
```



#### (B) Data Preprocessing
* Please use ResNet101 extract features of coco, flickr30k dataset and save to coco_precompute.hdf5 and f30k_precompute.hdf5:
```
pre() in utils/precompute.py
```
* Please use mini-batch KMeans algorithm clustering MSCOCO to get KMeans model 'coco_kmeans.pkl' and labels 'coco_labels.pkl', using :
```
clustering.py
```

Before running the code, please prepare the above data under the './data' folder.
```
- data
 - coco
  - coco_precompute.hdf5
  - coco_kmeans.pkl
  - coco-label.pkl
 - f30k
  - f30k_precompute.hdf5
```




### 3. Testing
Download mpnet (language feature extractor) and our pretrained model from:
[Baidu Wangpan](https://pan.baidu.com/s/1kG_aJH3a-ZufcEPfRTpBQg) (pwd:kpho)


Please put our provided pretrained model coco-f30k-cda2n-checkpoint.ckpt under the './checkpoints' folder and run:
```
cd itr
python rankretrival_circum.py --c_path /checkpoints/coco-f30k-cda2n-checkpoint.ckpt --test
```

