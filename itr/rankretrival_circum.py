import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
import argparse
import _pickle as cPickle
import time

import numpy as np
import torch, os
from torch.optim.lr_scheduler import StepLR
from transformers import AutoModel
import pytorch_lightning as pl
from torchvision.models import resnet101
import torch.nn as nn
from evaluation import i2t, ContrastiveLoss, t2i, RankerLoss
from baseline import l2norm
from dataset import precompute_image, precompute
import torch.utils.data as data
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
from pytorch_lightning import  loggers as pl_loggers
from dataset import WrapLoader
from torch.utils.data import DataLoader
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
import torch.nn.functional as F
from itr.coreset import active_sample
from itr.coreset import NLLLoss, FreeEnergyAlignmentLoss
from memory_profiler import profile
from torch.autograd import Variable
from copy import copy as copy
from copy import deepcopy as deepcopy
from scipy import stats
import pdb


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = '8'
from torch.autograd import Function


def init_centers(X, K=1000):
    embs = torch.Tensor(X)
    ind = torch.argmax(torch.norm(embs, 2, 1)).item() # 
    embs = embs.cuda()
    mu = [embs[ind]] # 
    indsAll = [ind] # 
    centInds = [0.] * len(embs) # 
    cent = 0
    print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1: #
            D2 = torch.cdist(mu[-1].view(1,-1), embs, 2)[0].cpu().numpy() # 
        else:
            newD = torch.cdist(mu[-1].view(1,-1), embs, 2)[0].cpu().numpy() #
            for i in range(len(embs)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent #
                    D2[i] = newD[i] #
        print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0: pdb.set_trace() # 
        D2 = D2.ravel().astype(float) # 
        Ddist = (D2 ** 2)/ sum(D2 ** 2) # 
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist)) # 
        ind = customDist.rvs(size=1)[0] # 
        while ind in indsAll: ind = customDist.rvs(size=1)[0] # 
        mu.append(embs[ind]) # 
        indsAll.append(ind) # 
        cent += 1 #
    return indsAll



class GradReverse(Function):

    @ staticmethod
    def forward(ctx, x, lambd, **kwargs: None):
        #　其实就是传入dict{'lambd' = lambd}
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, *grad_output):
        # 传入的是tuple，我们只需要第一个
        return grad_output[0] * -ctx.lambd, None

    def backward(ctx, grad_output):
        # 直接传入一格数
        return grad_output * -ctx.lambd, None
class retrivalactive(pl.LightningModule):
    def __init__(self,cluster_model,args,queried=None):
        super().__init__()
        # self.total_steps=total_steps
        self.clusterk=args.clusterk
        self.cluster_model=cluster_model
        self.lr=args.lr
        self.strategy=args.strategy
        self.rank=args.rank
        self.round=args.round
        resnet=resnet101(pretrained=True)
        resnet.fc=nn.Sequential()
        self.extractor=resnet
        self.embed=nn.Linear(2048,768)
        self.sentence_model=AutoModel.from_pretrained('./models')
        self.sentence_embed=nn.Linear(768,768)
        self.cross_entropy=nn.CrossEntropyLoss()
        self.cross_entropy_none=nn.CrossEntropyLoss(reduction='none')
        self.ranker=nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            nn.Softmax(dim=-1)
        )
        self.mse=nn.MSELoss(reduction="none")
        self.Logsoftmax=nn.LogSoftmax(dim=1)
        self.rankerloss=RankerLoss()
        self.criterion = ContrastiveLoss(margin=0.2,
                                         measure="cosine",
                                         max_violation=False, none_red=True)
        self.domain_classifier=nn.Sequential(nn.Linear(768+1,512),nn.ReLU(),nn.Linear(512,512),nn.ReLU(),nn.Linear(512,2))
        self.source_classifier=nn.Linear(768,512)
        self.prototypes=torch.tensor(cluster_model.cluster_centers_)
        # self.queried_captions=None
        self.softmax=nn.Softmax(dim=1)
        self.query_random_loader=None
        self.queries=[]
        self.queried_input_ids=[]
        self.queried_att_mask=[]
        self.queried_loader=None
        self.UL_idxs=None
        if queried is not None:   
            with open("kmeans_buffers_pre.pkl","rb") as f:
                buffers= cPickle.load(f)
            self.queries=buffers["queries"][:500*self.round]
            self.queried_input_ids=buffers["queried_input_ids"][:500*self.round]
            # self.queried_att_mask=buffers["queried_att_mask"]
        # self.peseudo_captions_idxs=None
        # self.peseudo_image_idxs=None
        self.PLT_dataset=None
        self.train_iter=None
        self.weight=0.1
        self.s0=args.s
        self.buget=1000
        self.warmup=30
        hparams={"s0":self.s0,"weight":self.weight,"strategy":self.strategy,"c_path":args.c_path}
        self.save_hyperparameters(hparams)
        self.best_val_acc=0
        self.is_query=True
        self.es=0
        self.save_k=0
        #self.model = model

        if not args.finetune:
            self.freeze_layer(self.extractor)
        self.freeze_layer(self.sentence_model)
        # self.init_weights()

    def early_stop(self,val_score):
        images=[]
        texts=[]
        score=val_score
        # score=6000.0
        if self.current_epoch==0:
            self.best_val_acc=score
            self.es=0
        if score > self.best_val_acc+0.5:
            self.best_val_acc = score
            self.es = 0

        else:
            self.es += 1

        n = 4
        if self.es > n: # overfitting
            self.is_query = True
            self.es = 0
            self.save_k = 10
            # self.save_k = 10

    def get_grad_embedding(self, X, Y,model=[]):
        #if type(model) == list:
        #    model = self.clf 
        
        embDim = 768 #model.get_embedding_dim()
        model.eval()
        nLab = len(np.unique(Y)) #标签数量
        embedding = np.zeros([len(Y), embDim * nLab])
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
                            shuffle=False, **self.args['loader_te_args'])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                cout, out = model(x)
                out = out.data.cpu().numpy()
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy() #
                maxInds = np.argmax(batchProbs,1) #
                for j in range(len(y)):
                    for c in range(nLab): 
                        if c == maxInds[j]: # 
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c]) # 将该类别的嵌入向量更新为输出向量 out[j] 的每个元素乘以（1 - 对应类别的概率）
                        else:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c]) # 将该类别的嵌入向量更新为输出向量 out[j] 的每个元素乘以（-1 * 对应类别的概率）
            return torch.Tensor(embedding)
        




          
    def active_ranker(self):
        image_dataset=precompute_image("train","f30k",return_idx=True)
        if len(self.queries) >0:
            indices=np.delete(np.arange(len(image_dataset)),self.queries)
            sampler=data.SubsetRandomSampler(indices)
            dataloader=data.DataLoader(image_dataset,batch_size=1000,sampler=sampler)
        else:
            dataloader=data.DataLoader(image_dataset,batch_size=1000)
        cluster_target=MiniBatchKMeans(n_clusters=self.buget//self.clusterk, random_state=0)
        rankers=[]
        UL_idxs=[]
        un_X=[]
        for idx,batch in enumerate(tqdm(dataloader)):
            images,_,sample_idx=batch
            images=images.to(self.device)
            with torch.no_grad():
                X=self.forward_image(images)
                r = self.ranker(X)
                logits = self.domain_classifier(torch.cat((X, r), -1))
                v_logits = self.softmax(self.source_classifier(X))
                d=self.softmax(logits)[:,-1]+torch.max(v_logits,-1).values#为1的概率（source）
                X=X.cpu().numpy()
                un_X.append(X)
                rankers.append(d.detach().cpu().numpy())
                UL_idxs.append(sample_idx.numpy())
        rankers=np.concatenate(rankers,axis=0)
        un_X=np.concatenate(un_X,0)
        UL_idxs=np.concatenate(UL_idxs,axis=0).astype(int)
        rank_ind = np.argsort(rankers)
        #去除阈值以上的
        kn_idxs=UL_idxs[rankers>1]
        kn_rankers=rankers[rankers>1]
        un_X=un_X[rankers<=1]
        UL_idxs=UL_idxs[rankers<=1]
        rankers=rankers[rankers<=1]
        rank_ind = np.argsort(rankers)
        cluster_target.fit(un_X)
        all_labels=cluster_target.labels_
        unique, counts = np.unique(all_labels, return_counts=True)
        cluster_prob = [0 for _ in range(cluster_target.n_clusters)]
        for index,label in enumerate(unique):

            cluster_prob[label]=counts[index]/sum(counts)
        #对ranker排序
        new_batch_cluster_counts = [0 for _ in range(cluster_target.n_clusters)]

        new_batch = [] #选择的batch
        for i in rank_ind:
            if len(new_batch) == self.buget:
                break
            #当前样本的label
            label = all_labels[i]
            #当前这个簇已选点所占百分比小于当前簇占所有点的百分比
            if new_batch_cluster_counts[label] / self.buget < cluster_prob[label]:
                new_batch.append(i)
                new_batch_cluster_counts[label] += 1
        n_slot_remaining = self.buget - len(new_batch)
        active_samples = [UL_idxs[i] for i in new_batch]
        kn_ind=np.argsort(kn_rankers)
        batch_filler = list(set(kn_ind))[0:n_slot_remaining]
        active_samples.extend([kn_idxs[i] for i in batch_filler])
        self.queries.extend(active_samples)
        self.queried_input_ids.extend([image_dataset.__getitem__(i)[1] for i in active_samples])
        self.log("query nums",len(self.queries))
       

        







    

#-------------------------------------------------------------------------------------------------------------------

    def get_simi_captions(self,embed,idx): # 获取与给定embedding和idx相似的caption
        #embed:(B,K)
        #idx:B or (B,1)
        idx_temp=np.array(idx,dtype=int) # 将输入索引转换为numpy数组
        # cps=torch.zeros_like(embed).type_as(embed)
        queries=[[i,np.where(self.queries==idx_temp[i]//5)[0][0]] for i in range(len(idx_temp)) if idx_temp[i]//5 in self.queries] # 查询并加入queries
        queries=np.asarray(queries)
        if len(queries)>0:
            #在self.queries的idx
            id=queries[:,1]
            input_ids=torch.stack(self.queried_input_ids)[id].to(self.device)
            # att_mask=torch.stack(self.queried_att_mask)[id].to(self.device)
            cps=self.forward_sentence(input_ids) 
            return cps,np.array(queries[:,0],dtype=int)
        return None,None



    def UL_images_simi(self,image_indexs,dataset):
        images=dataset[image_indexs]
        im_embed=self.embed(images.to(self.device)).detach().cpu()
        im_embed=l2norm(im_embed)
        # print(self.prototypes.shape,self.queried_captions.shape)
        captions=l2norm(torch.cat((self.prototypes,self.queried_captions)))
        simi=im_embed.view(-1,768).mm(captions.view(-1,768).t())
        simi=F.softmax(simi,dim=0)*F.softmax(simi,dim=1)
        indexs=torch.argmax(simi,dim=-1)
        return indexs


    def get_transfer_score(self,image_embed):
        with torch.no_grad():
            domain_logits=self.domain_classifier(image_embed)
            domain_logits=self.softmax(domain_logits)
            v_logits=self.source_classifier(image_embed).detach()
            v_logits=self.softmax(v_logits)
        v_logits=torch.max(v_logits,-1).values
        transfer_score=v_logits+domain_logits[:,-1]
        labels=torch.zeros(image_embed.shape[0]).long()
        labels[transfer_score>self.s0]=1
        return labels
    
    def cal_transfer_score(self,domain_prob,classify_prob):
        # print(classify_prob)
        v_logits=torch.max(self.softmax(classify_prob),-1).values
        transfer_score=v_logits+self.softmax(domain_prob)[:,1]
        labels=torch.zeros(domain_prob.shape[0]).long()
        labels[transfer_score>self.s0]=1 # 
        # 0 for target domain and 1 for source domain
        return transfer_score,labels
    

    def freeze_layer(self,layer):
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = False


    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        for layer in self.embed:
            if isinstance(layer,nn.Linear):
                r = np.sqrt(6.) / np.sqrt(layer.in_features +
                                          layer.out_features)
                layer.weight.data.uniform_(-r, r)
                layer.bias.data.fill_(0)


    def forward_image(self,images):
        if len(images.shape)>3:
            images=self.extractor(images)
        X=self.embed(images)
        X=l2norm(X)
        return X
    
    
    def forward_sentence(self,input_ids,att_mask=None):
        if att_mask is None:
            s_cp=self.sentence_embed(input_ids)
            s_cp=l2norm(s_cp)
            return s_cp
        s_cp=self.sentence_model(input_ids,att_mask)
        s_cp=self.mean_pooling(s_cp,att_mask)
        s_cp=self.sentence_embed(s_cp)
        s_cp=l2norm(s_cp)
        return s_cp
   
   
    def forward(self,s_im,s_input_ids,s_att_mask=None,t_im=None,t_idxs=None) :
        s_im=self.forward_image(s_im)
        if s_att_mask is None:
            s_cp=self.forward_sentence(s_input_ids)
        else:
            s_cp = self.forward_sentence(s_input_ids, s_att_mask)

        s_rank = torch.rand((s_im.shape[0], 1)).type_as(s_im)

        ranker_loss = 0
        # if self.current_epoch>0:
        s_im_re = GradReverse.apply(s_im, 1.0) 
        s_rank = self.ranker(s_im_re) # 
        s_cp_re = GradReverse.apply(s_cp, 1.0) # 
        s_im_dlogits = self.domain_classifier(torch.cat((s_im_re, s_rank.detach()), -1)) # D(r,Ev(x)): 连接s_im_re和s_rank,计算domain_logits
        s_cp_dlogits = self.domain_classifier(torch.cat((s_cp_re, s_rank.detach()), -1))
        s_im_vlogits = self.source_classifier(s_im) # C(Ev(x))
        s_cp_vlogits = self.source_classifier(s_cp) # C(Ev(y))
        if t_im is not None: # t_im next_itr中获取的数据
            # print(t_im)
            t_im = self.forward_image(t_im)
            # t_rank= torch.rand((t_im.shape[0],1)).type_as(s_im)
            # if self.current_epoch>0:
            t_im_re = GradReverse.apply(t_im, 1.0)
            t_rank = self.ranker(t_im_re)
            t_im_dlogits = self.domain_classifier(torch.cat((t_im_re, t_rank.detach()), -1))
            t_im_vlogits = self.source_classifier(t_im)
            if t_idxs is None:
                return s_im, t_im, s_cp, (s_im_dlogits, s_cp_dlogits, t_im_dlogits), (
                s_im_vlogits, s_cp_vlogits, t_im_vlogits), s_rank, t_rank
            else:
                t_cp, idxs = self.get_simi_captions(t_im, t_idxs)
                UL_idxs = np.arange(len(t_im))
                if t_cp is not None:
                    t_cp = t_cp.view(-1, 768)
                    t_cp = l2norm(t_cp)
                    t_cp_re = GradReverse.apply(t_cp, 1.0)
                    try:
                        t_cp_dlogits = self.domain_classifier(torch.cat((t_cp_re, t_rank[idxs].detach()), -1))
                    except:
                        print(t_im.shape,t_idxs.shape)
                        print(t_cp.shape,idxs.shape)
                        print(t_cp_re.shape,t_rank[idxs].shape)
                        exit(0)
                    t_cp_vlogits=self.source_classifier(t_cp)
                    UL_idxs=np.delete(np.arange(len(t_im)),idxs)
                else:
                    t_cp_dlogits=None
                    t_cp_vlogits=None
                return s_im,t_im,s_cp,t_cp,(s_im_dlogits,s_cp_dlogits,t_im_dlogits,t_cp_dlogits),(s_im_vlogits,s_cp_vlogits,t_im_vlogits,t_cp_vlogits),idxs,UL_idxs,s_rank,t_rank
        else:
            return s_im,s_cp,s_rank
    def mean_pooling(self,model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # def on_train_start(self):
    #     self.logger.log_hyperparams(self.hparams, {"hp/metric_1": 0, "hp/metric_2": 0})
    #     self.logger
    
    
    @profile
    def on_train_epoch_start(self):

        self.train_steps = args.max_epochs * self.train_dataloader().__len__()
        batch_size = self.train_dataloader().batch_size
        self.PLT_dataset=precompute("train","f30k",return_idx=True)
        self.train_iter=iter(DataLoader(dataset=self.PLT_dataset,batch_size=batch_size,shuffle=True,num_workers=16)) #数据迭代器（iterator），用于迭代训练数据集
        if len(self.queries)>=15000 and self.is_query:
            exit(0)
        if self.strategy=="rank" and len(self.queries)<15000 and self.is_query:
            self.is_query=False
            self.save_k=0
            self.active_ranker()
        elif self.strategy=="LP" and len(self.queries)<15000 and self.is_query:
            self.is_query=False
            self.save_k=0
            self.active_LP()
        elif self.strategy=="coreset" and len(self.queries)<15000 and self.is_query:
            self.is_query=False
            self.save_k=0
            self.active_coreset()
        elif self.strategy=="badge" and len(self.queries)<15000 and self.is_query:
            self.is_query=False
            self.save_k=0
            self.active_badge()
        elif self.strategy=="eada" and len(self.queries)<15000 and self.is_query:
            self.is_query=False
            self.save_k=0
            self.active_eada()

        input_ids=torch.stack(self.queried_input_ids)
            # att_mask=torch.stack(self.queried_att_mask)
            # dataset = data.Dataset(input_ids)
        self.queried_loader = DataLoader(input_ids, batch_size=5,shuffle=False) #5000
        if self.strategy=="warmup":#
            return; #



    ''',optimizer_idx'''
    @profile
    def training_step(self, batch, batch_idx) :
        step_percent = self.global_step / self.train_steps
        s_alpha = (2 + self.s0) / 2 - step_percent * (1 - self.s0 / 2) # s_alpha从1.5逐渐减小到0.5
        s_beta=self.s0/2+step_percent*self.s0/2 # s_beta从0.5逐渐减小到1
        self.log("params/aplha",s_alpha)
        self.log("params/beta",s_beta)
        if len(batch)==4:
            s_im,s_input_ids,s_att_mask,s_label=batch
        else:
            s_im,s_input_ids,s_label=batch
            s_att_mask=None
        # s_im,s_input_ids,s_att_mask,_=batch
        try:
            t_im,_,t_idxs=next(self.train_iter)
        except:
            batch_size=self.train_dataloader().batch_size
            self.train_iter=iter(DataLoader(dataset=self.PLT_dataset,batch_size=batch_size,shuffle=True))
            t_im,_,t_idxs=next(self.train_iter) #从 self.train_iter 迭代器中获取下一个数据批次，并将其分解为三个变量：t_im、_、t_idxs
        t_im=t_im.type_as(s_im) #数据类型转换
        s_im,t_im,s_cp,t_cp,(s_im_dlogits,s_cp_dlogits,t_im_dlogits,t_cp_dlogits),(s_im_vlogits,s_cp_vlogits,t_im_vlogits,t_cp_vlogits),idxs,UL_idxs,s_rank,t_rank=self.forward(s_im,s_input_ids,s_att_mask,t_im,t_idxs=t_idxs)
        # s_transfer_score,_=self.cal_transfer_score(s_im_dlogits,classify_prob=s_im_vlogits)
        t_tscore, _ = self.cal_transfer_score(t_im_dlogits, classify_prob=t_im_vlogits)
        # s_dlogits=s_im_dlogits
        s_dlogits = torch.cat((s_im_dlogits, s_cp_dlogits), dim=0) # source domain logits
        t_dlogits = t_im_dlogits # target domain logits
        if t_cp is not None:
            t_tscore2, _ = self.cal_transfer_score(t_cp_dlogits, classify_prob=t_cp_vlogits)
            s_dlogits = torch.cat((s_dlogits, t_cp_dlogits), dim=0)
            s_dlogits = torch.cat((s_dlogits, t_im_dlogits[idxs]), dim=0)
            t_dlogits = t_im_dlogits[UL_idxs]
        # transfer score 越高代表 是source domain的
        da_loss1 = self.cross_entropy(s_dlogits, torch.ones(s_dlogits.shape[0]).type_as(s_dlogits).long()) # source domain, 目标标签设置为1
        da_loss2 = self.cross_entropy(t_dlogits, torch.zeros(t_dlogits.shape[0]).to(self.device).long()) # target domain, 目标标签设置为0
        s_vlogits = torch.cat((s_im_vlogits, s_cp_vlogits), dim=0) # vlogits是由concept classifier得到的分数
        s_labels = torch.cat((s_label, s_label))
        C_loss = self.cross_entropy(s_vlogits, s_labels)

        CR_loss = torch.tensor(0).type_as(C_loss) # 初始化
        if len(t_tscore[t_tscore > s_alpha]) > 0: # t_tscore:transfer score, w(x)>w_alpha, select y from sentences in source domain
            # 属于source domain
            logits = t_im_vlogits[t_tscore > s_alpha]
            if t_cp is not None:
                logits2 = t_cp_vlogits[t_tscore2 > s_alpha]
                logits=torch.cat((logits,logits2),dim=0)
            t_labels = torch.argmax(self.softmax(logits), dim=-1)
            C_loss = self.cross_entropy(logits, t_labels)
        self.log("C_loss",C_loss)
        if len(t_tscore[t_tscore<s_beta])>0:
            #属于target domain
            logits=t_im_vlogits[t_tscore<s_beta]
            if t_cp is not None:
                logits2=t_cp_vlogits[t_tscore2<s_beta]
                logits=torch.cat((logits,logits2),dim=0)
            # print(logits.shape)
            prob=self.softmax(logits)
            mean_prob=torch.sum(prob,dim=0)/prob.shape[0]
            CR_loss=torch.sum(mean_prob*mean_prob)
            self.log("CR_loss",CR_loss)
        source_ranking_loss,source_ranking = self.criterion(s_im,s_cp)
        UL_score=t_tscore[UL_idxs]
        UL_im = t_im[UL_idxs]
        Pl_ranking_loss = torch.tensor(0).type_as(C_loss)
        if self.strategy=="rank":
            if len(UL_im[UL_score>s_alpha])>0:
                prob=self.softmax(t_im_vlogits)[UL_idxs]
                prob=prob[UL_score>s_alpha]
                cp_embeds=self.sentence_embed(self.prototypes.type_as(prob))
                cps=torch.einsum("ij,jk->ik",prob,cp_embeds)
                Pl_ranking_loss,_ = self.criterion(UL_im[UL_score>s_alpha],cps)
                self.log('Pl_ranking_loss', Pl_ranking_loss)
            if len(UL_im[UL_score<s_beta])>0 and self.queried_loader is not None and self.strategy !="random":
                ims=UL_im[UL_score<s_beta]
                simis=[]
                all_cps=[]
                for b_idx,sample in enumerate(self.queried_loader):
                    # print(sample.shape)
                    if len(sample.shape)==2:
                        sample=sample.to(self.device)
                        batch_cps=self.forward_sentence(sample)
                    else:
                        input_ids,att_mask=sample
                        input_ids,att_mask=input_ids.to(self.device),att_mask.to(self.device)
                        batch_cps=self.forward_sentence(input_ids,att_mask)
                    simi=torch.einsum("ij,jk->ik",ims,batch_cps.t())
                    simis.append(simi)
                    all_cps.append(batch_cps)
                simis=torch.cat(simis,-1)
                simis=F.softmax(simis,0)*F.softmax(simis,1)
                rank_id=np.argmax(simis.cpu().detach().numpy(),-1)
                cps=torch.cat(all_cps,0)[rank_id]
                Pl_ranking_loss,_ = self.criterion(ims,cps)
                self.log('Pl_ranking_loss', Pl_ranking_loss)
        target_ranking_loss = torch.tensor(0).type_as(C_loss)
        if t_cp is not None:
            target_ranking_loss,target_ranking = self.criterion(t_im[idxs],t_cp)
            self.log('target_ranking_loss', target_ranking_loss)
        if self.rank=="CE":
            ranker_loss=self.rankerloss(s_rank,self.cross_entropy_none(s_im_vlogits,s_label))
        elif self.rank=="ranking":
            ranker_loss=self.rankerloss(s_rank,source_ranking)
            if t_cp is not None and len(t_cp)>1:
                ranker_loss=self.rankerloss(t_rank[idxs],target_ranking)
                self.log('ranker_loss', ranker_loss)

        if self.strategy=="LP":
            loss = C_loss + CR_loss + source_ranking_loss + target_ranking_loss + da_loss2 + da_loss1
        else:
            loss = C_loss + CR_loss + source_ranking_loss + target_ranking_loss + Pl_ranking_loss + da_loss2 + ranker_loss + da_loss1
        #     #update feature extractor and classifier,to fool discrimiator
        #     loss=da_loss1+C_loss+CR_loss+source_ranking_loss+target_ranking_loss+Pl_ranking_loss
        #     # self.log('train_loss', loss)
        # else:
        #     #update discrimiator
        #     loss=da_loss2
        values = {
            "da_loss1": da_loss1,
            "da_loss2": da_loss2,
            "source_ranking_loss": source_ranking_loss,
            # "ranker_loss":ranker_loss
        }

        if torch.isnan(loss):
            print(values)
            print(ranker_loss)
            print(C_loss)
            print(CR_loss)
            exit(0)
        self.log_dict(values)
        # exit(0)
        return loss

    #TODO 更新prototypes
    #def training_epoch_end():
    def validation_step(self, batch,batch_idx,dataloader_idx):
        # print("idx",dataloader_idx)
        if dataloader_idx==0:
            if len(batch)>2:
                im,input_ids,att_mask=batch
                im,s,rank=self(im,input_ids,att_mask)
            else:
                im,sent=batch
                im,s,rank=self(im,sent)
            return {"simages":im,"scaptions":s,"rank":rank}
        else:
            if len(batch)>2:
                im,input_ids,att_mask=batch
                im,s,_=self(im,input_ids,att_mask)
            else:
                im,sent=batch
                im,s,_=self(im,sent)
            return {"timages":im,"tcaptions":s}


    def validation_epoch_end(self, outputs):
        currscore=0
        simages=[]
        scaptions=[]
        timages=[]
        tcaptions=[]
        sranks=[]
        #outputs=[[dataloader0_outputs{},{}],[dataloader1_ouputs]]
        for outs in outputs:
            #outs=[{},{}]
            if len(outs)==0:
                continue
            for output in outs:
                #output:{}
                # print(output)
                if "simages" in output:
                    simages.append(output["simages"])
                    scaptions.append(output["scaptions"])
                    sranks.append(output["rank"])
                if "timages" in output:
                    # if self.current_epoch>=self.warmup:
                    timages.append(output["timages"])
                    tcaptions.append(output["tcaptions"])
        if len(simages)>0:
            simages=torch.cat(simages,0)
            scaptions=torch.cat(scaptions,0)
            sranks=torch.cat(sranks,0)
            if self.strategy=="warmup":
                _,ranking=self.criterion(simages,scaptions)
                ranker_loss=self.rankerloss(sranks,ranking)
            simages=simages.cpu().numpy()
            scaptions=scaptions.cpu().numpy()
            # exit(0)
            (i2t_r1, i2t_r5, i2t_r10, i2t_medr, i2t_meanr) = i2t(simages, scaptions,per=5)
            (t2i_r1, t2i_r5, t2i_r10, t2i_medr, t2i_meanr) = t2i(scaptions, scaptions,per=5)

            currscore_s = i2t_r1 + i2t_r5 + i2t_r10 + t2i_r1 + t2i_r5 + t2i_r10
            currscore=currscore_s
            values_s = {"source/currscore":currscore_s,'source/i2t_r1': i2t_r1, 'source/i2t_r5': i2t_r5, 'source/i2t_r10': i2t_r10,'source/i2t_medr':i2t_medr,'source/i2t_meanr':i2t_meanr,'source/t2i_r1':t2i_r1,'source/t2i_r5':t2i_r5,'source/t2i_r10':t2i_r10,'source/t2i_medr':t2i_medr,'source/t2i_meanr':t2i_meanr}
            self.log_dict(values_s)
            self.log("currscore_s",currscore_s)
            if self.strategy in ["rank","LP", "eada","badge"] : #new
                self.early_stop(currscore_s)
                self.log("save_k",self.save_k)

        # print(len(timages))
        if len(timages)>0:
            timages = torch.cat(timages, 0)
            tcaptions = torch.cat(tcaptions, 0)
            timages = timages.cpu().numpy()
            tcaptions = tcaptions.cpu().numpy()
            (i2t_r1, i2t_r5, i2t_r10, i2t_medr, i2t_meanr) = i2t(timages, tcaptions, per=5)
            (t2i_r1, t2i_r5, t2i_r10, t2i_medr, t2i_meanr) = t2i(timages, tcaptions, per=5)
            currscore_t = i2t_r1 + i2t_r5 + i2t_r10 + t2i_r1 + t2i_r5 + t2i_r10
            values_t = {"currscore_t": currscore_t, 'target/i2t_r1': i2t_r1, 'target/i2t_r5': i2t_r5,
                        'target/i2t_r10': i2t_r10, 'target/i2t_medr': i2t_medr, 'target/i2t_meanr': i2t_meanr,
                        'target/t2i_r1': t2i_r1, 'target/t2i_r5': t2i_r5, 'target/t2i_r10': t2i_r10,
                        'target/t2i_medr': t2i_medr, 'target/t2i_meanr': t2i_meanr}
            self.log_dict(values_t)
            if self.strategy == "warmup":
                self.log("save_k", currscore_t)
        return currscore
    def test_step(self, batch, batch_idx):
        if len(batch)>2:
            t_im,t_input_ids,t_att_mask=batch
            im,s,_=self(t_im,t_input_ids,t_att_mask)
        else:
            im,sent=batch
            im,s,_=self(im,sent)
        return {"images":im,"captions":s}

    def test_epoch_end(self, outputs):
        img_embs=[]
        cap_embs=[]
        for output in outputs:
            # print(output)
            img_embs.append(output["images"])
            cap_embs.append(output["captions"])
        img_embs=torch.cat(img_embs,0)
        cap_embs=torch.cat(cap_embs,0)
        img_embs=img_embs.cpu().numpy()
        cap_embs=cap_embs.cpu().numpy()
        # print(img_embs.shape,cap_embs.shape)
        # (i2t_r1, i2t_r5, i2t_r10, i2t_medr, i2t_meanr), (t2i_r1, t2i_r5, t2i_r10, t2i_medr, t2i_meanr) = i2t(img_embs, cap_embs, model=self)
        # caption retrieval
        (i2t_r1, i2t_r5, i2t_r10, i2t_medr, i2t_meanr) = i2t(img_embs, cap_embs,per=5)
        # image retrieval
        (t2i_r1, t2i_r5, t2i_r10, t2i_medr, t2i_meanr) = t2i(img_embs, cap_embs,per=5)
        # sum of recalls to be used for early stopping
        currscore = i2t_r1 + i2t_r5 + i2t_r10 + t2i_r1 + t2i_r5 + t2i_r10
        values = {'i2t_r1': i2t_r1, 'i2t_r5': i2t_r5, 'i2t_r10': i2t_r10,'i2t_medr':i2t_medr,'i2t_meanr':i2t_meanr,'t2i_r1':t2i_r1,'t2i_r5':t2i_r5,'t2i_r10':t2i_r10,'t2i_medr':t2i_medr,'t2i_meanr':t2i_meanr}
        print(values)
        print("currscore:",currscore)
        return currscore

    def configure_optimizers(self):

        ranker_params=self.ranker.parameters()
        params = list(self.embed.parameters()) + list(self.sentence_embed.parameters()) + list(
            self.source_classifier.parameters()) + list(self.domain_classifier.parameters())

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.95)  # 5,0.9
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

def train(args):
    save_dir=args.save_dir
    dataset = "coco-f30k-precompute"
    dataset = args.dataset
    cluster_model = cPickle.load(open("./data/coco/coco_kmeans.pkl", "rb"))
    checkpoint_callback = ModelCheckpoint(
        monitor='save_k',
        dirpath=save_dir,
        filename ="ranker/"+dataset+"-"+args.strategy+"-"+args.rank+str(args.clusterk)+"s"+str(args.s)+'-{epoch:02d}-{currscore_s:.2f}-{currscore_t:.2f}',
        # save_top_k = 16,
        save_top_k = 3,
        save_last=True,
        mode = 'max')
    # checkpoint_callback = ModelCheckpoint(
    #     monitor='currscore_s',
    #     dirpath=save_dir,
    #     filename ="ranker/"+dataset+"-"+args.strategy+"-"+args.rank+'-{epoch:02d}-{currscore_s:.2f}',
    #     save_top_k = 3,
    #     save_last=True,
    #     mode = 'max')
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=dataset + "/active/",
                                             name="ranker-""-" + args.strategy + args.rank)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    loader=WrapLoader(args=args,dataset=dataset)
    gpu_num=len(args.gpus.split(","))
    batch_sz = args.batch_sz * gpu_num
    train_epoch = args.max_epochs
    import math
    train_steps = train_epoch * math.ceil(82783 / batch_sz)

    selected=None

    if args.c_path!="":
        model=retrivalactive.load_from_checkpoint(checkpoint_path=args.c_path,strict=False,cluster_model=cluster_model,args=args,queried=selected)
    else:
        model=retrivalactive(cluster_model=cluster_model,args=args,queried=selected)
    trainer = pl.Trainer(gradient_clip_val=2.0,gpus=args.gpus,log_every_n_steps=10,callbacks=[checkpoint_callback,lr_monitor],logger=tb_logger,max_epochs=args.max_epochs)
    loader.setup("fit")
    val_loader=loader.val_dataloader()  ##
    trainer.validate(model,val_dataloaders=val_loader)##
    trainer.fit(model, datamodule=loader)
    lr_finder = trainer.tuner.lr_find(model,datamodule=loader)##   
    new_lr = lr_finder.suggestion()##
    print(new_lr)##

def test(args):
    cluster_model = cPickle.load(open("./data/coco/coco_kmeans.pkl", "rb"))
    if args.c_path == "":
        model = retrivalactive(cluster_model=cluster_model, args=args)
    else:
        model = retrivalactive.load_from_checkpoint(checkpoint_path=args.c_path, cluster_model=cluster_model, args=args)
    trainer = pl.Trainer(gradient_clip_val=2.0, gpus=args.gpus, log_every_n_steps=10, max_epochs=120)
    loader = WrapLoader(args=args, dataset=args.dataset)
    start=time.time()
    trainer.test(model, datamodule=loader)
    end=time.time()
    print(end-start)
    # loader.setup("fit")
    # trainer.validate(model, datamodule=loader)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=5e-5,type=float)#
    parser.add_argument('--batch_sz', default=120,type=int)
    parser.add_argument('--num_workers',default=8,type=int)
    parser.add_argument('--gpus',default='4',type=str) # default='7'
    parser.add_argument('--test', action='store_true',default=True) # default=True
    parser.add_argument('--c_path',default="./checkpoints/f30k-epoch=23-currscore=360.81.ckpt",type=str)
    parser.add_argument('--finetune',action='store_true')
    parser.add_argument('--max_epochs',default=30,type=int)
    parser.add_argument('--strategy', default="ranker", type=str)
    parser.add_argument('--rank', default="ranking", type=str)
    parser.add_argument('--round',default=15,type=int)
    parser.add_argument('--clusterk',default=1,type=int)
    parser.add_argument('--s',default=1,type=float)
    #cluster nums =budget/K
    parser.add_argument('--dataset',default="coco-f30k-precompute",type=str)
    #parser.add_argument('--save_dir',default="adapt_checkpoints_3_rankdebug",type=str)
    parser.add_argument('--save_dir',default="adapt_checkpoints",type=str) 
    args = parser.parse_args()
    # train(args)
    if args.test:
        #train(args)
        test(args)

    else:
        train(args)