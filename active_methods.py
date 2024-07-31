#     # self.log("query_num",len(self.queries))
# def clustering(self):
#     #按一个batch来对未标注的target图像进聚类
#     image_dataset=self.PLT_dataset
#     if len(self.queries) >0:
#         indices=np.delete(np.arange(len(image_dataset)),self.queries)
#         sampler=data.SubsetRandomSampler(indices)
#         dataloader=data.DataLoader(image_dataset,batch_size=1000,sampler=sampler)
#     else:
#         dataloader=data.DataLoader(image_dataset,batch_size=1000)
#
#     cluster_target=MiniBatchKMeans(n_clusters=512, random_state=0,compute_labels=False,batch_size=1000)
#     for idx,batch in enumerate(dataloader):
#         images, _,_,_=batch
#         images=images.to(self.device)
#         with torch.no_grad():
#             X=self.forward_image(images).cpu().numpy()
#         cluster_target = cluster_target.partial_fit(X)
#         del X
#     #对聚类中心计算transfer score，返回1/0标签代表已知未知
#     labels=self.get_transfer_score(torch.tensor(cluster_target.cluster_centers_).to(self.device))
#     #选出unknown的中心
#     centers=cluster_target.cluster_centers_[labels.numpy()<1]
#     assert len(centers)>0
#     unknown_center_labels=np.arange(len(cluster_target.cluster_centers_))[labels.numpy()<1]
#     closests=np.empty(shape=(0,len(centers)),dtype=np.uint16)
#     dists=np.empty(shape=(0,len(centers)),dtype=np.float16)
#     #为每个聚类中心寻找最近的样本
#     bath_sz=dataloader.batch_size
#     UL_idxs=[]
#     UL_labels=[]
#     min_margins=[]
#
#     # for idx,batch in enumerate(dataloader):
#     #     images, _,_,sample_idx=batch
#     #     images=images.to(self.device)
#     #     with torch.no_grad():
#     #         X=self.forward_image(images).cpu().numpy()
#     #     #(batch_sz,)
#     #     batch_labels=cluster_target.predict(X)
#     #     UL_extend_idxs=[np.append(X[i,:],sample_idx[i]) for i in range(X.shape[0]) if batch_labels[i] in unknown_center_labels]
#     #     UL_extend_idxs=np.stack(UL_extend_idxs,0)
#     #     UL_idxs.extend(UL_extend_idxs[:,-1])
#     #     # try:
#     #     closest, distances = pairwise_distances_argmin_min(centers,UL_extend_idxs[:,:-1])
#     #     # except:
#     #     #
#     #     #     print(centers.shape,X.shape)
#     #     del X,batch
#     #     closests=np.concatenate((closests,(closest+idx*bath_sz).reshape(1,len(centers))),axis=0)
#     #     dists=np.concatenate((dists,distances.reshape(1,-1)),axis=0)
#     # del dataloader
#     # idxs=np.argmin(dists,axis=0)
#     # closest=[closests[idxs[i],i]for i in range(len(centers))]
#     # #增加新的query idx
#     # self.queries.extend(np.array(UL_idxs)[closest])
#     # self.queries=list(map(int,self.queries))
#     # closest=list(map(int,closest))
#     # self.UL_idxs=np.delete(np.array(UL_idxs),closest)
#     # self.queried_input_ids=[image_dataset.__getitem__(i)[1] for i in self.queries]
#     # self.queried_att_mask=[image_dataset.__getitem__(i)[2] for i in self.queries]
#     # self.queried_input_ids=torch.stack(self.queried_input_ids)
#     # self.queried_att_mask=torch.stack(self.queried_att_mask)
#     # self.log("query_num",len(self.queries))
#     # self.peseudo_captions_idxs=self.UL_images_simi(self.UL_images,image_dataset)
#     # assert len(self.peseudo_captions_idxs)==len(self.UL_images)
#     # exit(0)
#     # captions_total=precompute_captiontotal("train")
#     # self.pseudo_caption_idxs=self.similarity_ic(image_dataset,self.UL_images,captions_total,self.queries*10)
#     # exit(0)
#     # self.peseudo_image_idxs=np.concatenate((self.queries,self.UL_images))
#     # self.peseudo_captions=torch.cat((self.prototypes,self.queried_captions))
#     # del image_dataset

def get_simi_captions(self,embed,idx):
    #embed:(B,K)
    #idx:B or (B,1)
    idx_temp=np.array(idx)
    embed_sorted=[]
    cp=[]
    queries=[[i,np.where(self.queries==i)] for i in idx_temp if i in self.queries]
    if len(queries)>0:
        idxs=np.array(queries)
        idx_temp=np.delete(idx_temp,idxs[:,0])
        id=idxs[:,-1]
        cp1=self.forward_sentence(self.queried_input_ids[id].to(self.device),self.queried_att_mask[id].to(self.device))
        cp.append(cp1)
        embed_sorted.append(embed[idxs[:,0]])
    queries=[[i,np.where(self.UL_idxs==i)] for i in idx_temp if i in self.UL_idxs]
    if len(queries)>0:
        idxs=np.array(queries)
        idx_temp=np.delete(idx_temp,idxs[:,0])
        id=idxs[:,-1]
        query_embeds=embed[idxs[:,0]]
        embed_sorted.append(query_embeds)
        all_cps=self.forward_sentence(self.queried_input_ids.to(self.device),self.queried_att_mask.to(self.device))
        simi=query_embeds.view(-1,768).mm(all_cps.view(-1,768).t())
        simi=F.softmax(simi,dim=0)*F.softmax(simi,dim=1)
        index=torch.argmax(simi,dim=-1)
        cp2=all_cps[index]
        cp.append(cp2)
    queries=idx_temp
    if len(queries)>0:
        prototype_cps= self.sentence_embed(self.prototypes.to(self.device))
        cps=l2norm(prototype_cps)
        query_embeds=embed[queries]
        embed_sorted.append(embed[queries])
        simi=query_embeds.view(-1,768).mm(cps.view(-1,768).t())
        simi=F.softmax(simi,dim=0)*F.softmax(simi,dim=1)
        index=torch.argmax(simi,dim=-1)
        cp3=cps[index]
        cp.append(cp3)
    return torch.cat(embed_sorted,0),torch.cat(cp,0)