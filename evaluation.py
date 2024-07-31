import torch
from torch import nn
import numpy
from torch.autograd import Variable
def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X
def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score

class  ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False,none_red=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation
        self.none_red=none_red

    def forward(self, im, s): # 图像-文本匹配
        # compute image-sentence score matrix
        scores = self.sim(im, s) # 计算余弦相似度分数矩阵
        diagonal = scores.diag().view(im.size(0), 1) # 提取对角线元素，按照每个样本进行拓展
        d1 = diagonal.expand_as(scores) # 行方向
        d2 = diagonal.t().expand_as(scores) # 列方向

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0) # 图像-文本配对损失
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0) # 文本-图像配对损失

        # clear diagonals
        mask = torch.eye(scores.size(0)).type_as(scores)> .5
        # I = Variable(mask)
        # if torch.cuda.is_available():
        #     I = I.type_as(scores).bool()
        cost_s = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(mask, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        if self.none_red:
            return cost_s.sum() + cost_im.sum(), cost_im
        return cost_s.sum() + cost_im.sum()
class RankerLoss(nn.Module):
    def __init__(self, margin=1.0, reduction='mean'):
        super(RankerLoss, self).__init__()
        self.margin=margin
        self.reduction=reduction

    def forward(self,input, target):
        if len(input)%2!=0 and len(input)>2:
            input=input[:-1]
            target=target[:-1]
        assert len(input) % 2 == 0, 'the batch size is not even.'
        assert input.shape == input.flip(0).shape

        input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
        target = (target - target.flip(0))[:len(target)//2]
        target = target.detach()

        one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
        input=torch.sigmoid(input)
        if self.reduction == 'mean':
            # loss = torch.sum(torch.clamp(self.margin - one * input, min=0))
            # loss = loss / input.size(0) # Note that the size of input is already halved
            loss=-torch.sum(one*torch.log(input)+(1-one)*torch.log(1-input))
            loss = loss / input.size(0)
        elif self.reduction == 'none':
            loss = torch.clamp(self.margin - one * input, min=0)
        else:
            NotImplementedError()

        return loss


def i2t(images, captions, npts=None,per=5,measure='cosine', return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images, 每行为一个图像特征向量
    Captions: (5N, K) matrix of captions, 每行代表一个caption特征向量
    """
    if npts is None:
        npts = int(images.shape[0] / per) # 计算需要处理数据点的数量=输入图像行数/per
    index_list = []

    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)
    top5 = numpy.zeros((npts,5))
    for index in range(npts):

        # Get query image
        im = images[per * index].reshape(1, images.shape[1])

        # Compute scores
        d = numpy.dot(im, captions.T).flatten() # 计算查询图像与所有caption之间的相似度
        # if measure == 'order':
        #     bs = 100
        #     if index % bs == 0:
        #         mx = min(images.shape[0], 5 * (index + bs))
        #         im2 = images[5 * index:mx:5]
        #         d2 = order_sim(torch.Tensor(im2).cuda(),
        #                        torch.Tensor(captions).cuda())
        #         d2 = d2.cpu().numpy()
        #     d = d2[index % bs]
        # else:
        #     d = numpy.dot(im, captions.T).flatten()
        inds = numpy.argsort(d)[::-1] # 根据相似度对caption进行排序，返回索引数组
        index_list.append(inds[0]) # 将查询结果的第一个索引添加到列表中

        # Score
        rank = 1e20 # 初始化排名为一个较大的值
        #img对应的正确caption，的最小rank
        for i in range(per * index, per * index + per, 1): # 遍历每个查询图像对应的caption
            tmp = numpy.where(inds == i)[0][0] # 查找正确caption的索引
            if tmp < rank: # 如果当前caption索引的位置小于当前rank，说明找到了更好的rank
                rank = tmp
        ranks[index] = rank # 将查询图像的排名存储到排名数组中
        top1[index] = inds[0] # 将查询图像的top1索引存储到top1数组中
        top5[index,:]=inds[:5] #  将查询图像的top5索引存储到top5数组中

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks) # 计算R@1
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks) # 计算R@5
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks) # 计算R@10
    medr = numpy.floor(numpy.median(ranks)) + 1 # 计算中位数排名
    meanr = ranks.mean() + 1 
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1,top5)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, npts=None,per=5, measure='cosine', return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = int(images.shape[0] / per)
    ims = numpy.array([images[i] for i in range(0, len(images), per)])
    # print(ims.shape,images.shape)
    # exit(0)
    # print(ims.shape,images.shape)
    ranks = numpy.zeros(per * npts)
    top1 = numpy.zeros(per * npts)
    top5 = numpy.zeros((per * npts,5))
    for index in range(npts):

        # Get query captions
        queries = captions[per * index:per * index + per]

        # Compute scores
        d = numpy.dot(queries, ims.T)
        # if measure == 'order':
        #     bs = 100
        #     if 5 * index % bs == 0:
        #         mx = min(captions.shape[0], 5 * index + bs)
        #         q2 = captions[5 * index:mx]
        #         d2 = order_sim(torch.Tensor(ims).cuda(),
        #                        torch.Tensor(q2).cuda())
        #         d2 = d2.cpu().numpy()
        #
        #     d = d2[:, (5 * index) % bs:(5 * index) % bs + 5].T
        # else:
        #     d = numpy.dot(queries, ims.T)
        inds = numpy.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = numpy.argsort(d[i])[::-1]
            ranks[per * index + i] = numpy.where(inds[i] == index)[0][0]
            top1[per * index + i] = inds[i][0]
            top5[per * index + i,:]=inds[i][:5]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1, top5)
    else:
        return (r1, r5, r10, medr, meanr)

def i2tandt2i(images, captions,per=5, npts=None, return_ranks=False,model=None):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    measure with cosine
    """
    if npts is None:
        npts = images.shape[0] // per
    index_list = []
    # gv1_list = []
    # gv2_list = []

    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)
    top5 = numpy.zeros((npts,5))

    score_matrix = numpy.zeros((images.shape[0] // per, captions.shape[0]))

    for index in range(npts):

        # Get query image
        # im = images[5 * index].reshape(1, images.shape[1])

        # Compute scores

        bs = 5
        #每bs个计算一次
        if index % bs == 0:
            # print ('['+str(index)+'/'+str(npts)+']')
            mx = min(images.shape[0], per * (index + bs))
            #从5*index开始，每五个取一个，共取bs个
            im2 = images[per * index:mx:per]
            #计算这个batch每张图片和每个句子的相似度
            d2 =im2.mm(captions.t())
            d2 = d2.data.cpu().numpy()
        #每次取一个图像的相似度序列
        d = d2[index % bs]
        #从大到小排序
        inds = numpy.argsort(d)[::-1]
        #每张图片最相似的句子编号
        index_list.append(inds[0])
        #存储相似度矩阵
        score_matrix[index] = d

        # Score
        rank = 1e20
        for i in range(per * index, per * index + per, 1):
            #选择第i个句子的图像index
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]
        top5[index,:] = inds[:5]

    #i2t
    stat_num = 0
    #想找每个正确的句子在image选择的句子中的最小rank
    minnum_rank_image = numpy.array([1e7]*npts)
    for i in range(npts):
        #对于第i张图片的句子相似度排序index
        cur_rank = numpy.argsort(score_matrix[i])[::-1]
        minnum_rank_image[i]=numpy.min(cur_rank[per*i:per*i+per])
        # for index, j in enumerate(cur_rank):
        #     #第index个句子，
        #     #[5*i, 5*i+5]是i图像正确句子的范围
        #     if j in range(5*i, 5*i+5):
        #         #如果
        #         stat_num += 1
        #         minnum_rank_image[i] = index
        #         break
    # print ("i2t stat num:", stat_num)

    i2t_r1 = 100.0 * len(numpy.where(minnum_rank_image<1)[0]) / len(minnum_rank_image)
    i2t_r5 = 100.0 * len(numpy.where(minnum_rank_image<5)[0]) / len(minnum_rank_image)
    i2t_r10 = 100.0 * len(numpy.where(minnum_rank_image<10)[0]) / len(minnum_rank_image)
    i2t_medr = numpy.floor(numpy.median(minnum_rank_image)) + 1
    i2t_meanr = minnum_rank_image.mean() + 1

    #print("i2t results:", i2t_r1, i2t_r5, i2t_r10, i2t_medr, i2t_meanr)

    #t2i

    stat_num = 0
    score_matrix = score_matrix.transpose()
    minnum_rank_caption = numpy.array([1e7]*npts*5)
    for i in range(per*npts):
        img_id = i // per
        cur_rank = numpy.argsort(score_matrix[i])[::-1]
        minnum_rank_caption[i]=cur_rank[img_id]
        # for index, j in enumerate(cur_rank):
        #     if j == img_id:
        #         stat_num += 1
        #         minnum_rank_caption[i] = index
        #         break

    # print ("t2i stat num:", stat_num)

    t2i_r1 = 100.0 * len(numpy.where(minnum_rank_caption<1)[0]) / len(minnum_rank_caption)
    t2i_r5 = 100.0 * len(numpy.where(minnum_rank_caption<5)[0]) / len(minnum_rank_caption)
    t2i_r10 = 100.0 * len(numpy.where(minnum_rank_caption<10)[0]) / len(minnum_rank_caption)
    t2i_medr = numpy.floor(numpy.median(minnum_rank_caption)) + 1
    t2i_meanr = minnum_rank_caption.mean() + 1


    # print("t2i results:", t2i_r1, t2i_r5, t2i_r10, t2i_medr, t2i_meanr)

    # Compute metrics
    # r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    # r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    # r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    # medr = numpy.floor(numpy.median(ranks)) + 1
    # meanr = ranks.mean() + 1
    if return_ranks:
        return (i2t_r1, i2t_r5, i2t_r10, i2t_medr, i2t_meanr), (t2i_r1, t2i_r5, t2i_r10, t2i_medr, t2i_meanr), score_matrix
    else:
        return (i2t_r1, i2t_r5, i2t_r10, i2t_medr, i2t_meanr), (t2i_r1, t2i_r5, t2i_r10, t2i_medr, t2i_meanr)
import numpy as np
def eval(query_embs,retro_embs,query_ids,retro_ids):
    sims = query_embs.dot(retro_embs.T)
    inds = np.argsort(sims, axis=1)
    label_matrix = np.zeros(inds.shape)
    for index in range(inds.shape[0]):
        ind = inds[index][::-1]
        label_matrix[index][np.where(np.array(retro_ids)[ind]==query_ids[index])[0]]=1

    ranks = np.zeros(label_matrix.shape[0])

    for index in range(len(ranks)):
        rank = np.where(label_matrix[index]==1)[0] + 1
        ranks[index] = rank[0]

    r1, r5, r10 = [100.0*np.mean([x <= k for x in ranks]) for k in [1, 5, 10]]
    medr = np.floor(np.median(ranks))
    meanr = ranks.mean()
    mir = (1.0/ranks).mean()

    return r1, r5, r10, medr, meanr