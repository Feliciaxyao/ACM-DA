from __future__ import print_function, division
import numpy as np
import torch
from torchvision import transforms, utils
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.models as models
import pdb
from datetime import datetime
import argparse
import pprint
import time

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from sklearn.metrics import pairwise_distances

class Coreset_Greedy:
    def __init__(self, all_pts):
        self.all_pts = np.array(all_pts)
        print(all_pts.shape())
        self.dset_size = len(all_pts)
        self.min_distances = None
        self.already_selected = []

        # reshape
        feature_len = self.all_pts[0].shape[1]
        self.all_pts = self.all_pts.reshape(-1,feature_len)

        # self.first_time = True

    def update_dist(self, centers, only_new=True, reset_dist=False):
        if reset_dist:
            self.min_distances = None
        if only_new:
            centers = [p for p in centers if p not in self.already_selected]

        if centers is not None:
            x = self.all_pts[centers] # pick only centers
            dist = pairwise_distances(self.all_pts, x, metric='euclidean')

            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1,1)
            else:
                self.min_distances = np.minimum(self.min_distances, dist)

    def sample(self, already_selected, sample_size):

        # initially updating the distances
        self.update_dist(already_selected, only_new=False, reset_dist=True)
        self.already_selected = already_selected

        # epdb.set_trace()

        new_batch = []
        # pdb.set_trace()
        for _ in range(sample_size):
            if self.already_selected == []:
                ind = np.random.choice(np.arange(self.dset_size))
            else:
                ind = np.argmax(self.min_distances)

            assert ind not in already_selected
            self.update_dist([ind],only_new=True, reset_dist=False)
            new_batch.append(ind)

        max_distance = max(self.min_distances)
        print("Max distance from cluster : %0.2f" % max_distance)

        return new_batch, max_distance
def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


def active_sample(labeled_features,unlabeled_features,unlab_idxs,sample_size, method='random', model=None):

    if method == 'coreset':
        # get features
        
        all_features = labeled_features + unlabeled_features
        labeled_indices = np.arange(0,len(labeled_features))

        coreset = Coreset_Greedy(all_features)
        new_batch, max_distance = coreset.sample(labeled_indices, sample_size)

        # unlabeled rows start after labeled rows in all_features
        # so offset the indices
        print(len(labeled_features),len(all_features)-len(unlabeled_features),len(unlab_idxs))
        new_batch = [i - len(labeled_features) for i in new_batch]
        sample_rows = unlab_idxs[new_batch]

        return sample_rows

    # if method == 'dbal_bald':
    #     # according to BALD implementation by Riashat Islam
    #     # first randomly sample 2000 points
    #     dropout_pool_size = 2000
    #     unl_rows = np.copy(unlabeled_rows)
    #
    #     if len(unl_rows) >= dropout_pool_size:
    #         np.random.shuffle(unl_rows)
    #         dropout_pool = unl_rows[:dropout_pool_size]
    #         temp_unlabeled_csv = 'unlabeled_temp.csv'
    #         np.savetxt(os.path.join(args.dataset_root, temp_unlabeled_csv), dropout_pool,'%s,%s',delimiter=',')
    #         csv_file = temp_unlabeled_csv
    #     else:
    #         dropout_pool = unl_rows
    #         csv_file = 'unlabeled.csv'
    #
    #
    #
    #     #create unlabeled loader
    #     data_transforms = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.1307,), (0.3081,))
    #     ])
    #
    #     unlab_dset = MNIST(args.dataset_root, subset='train',csv_file=csv_file,transform=data_transforms)
    #     unlab_loader = DataLoader(unlab_dset, batch_size=1, shuffle=False, **kwargs)
    #
    #     scores_sum = np.zeros(shape=(len(dropout_pool), args.nclasses))
    #     entropy_sum = np.zeros(shape=(len(dropout_pool)))
    #
    #     for _ in range(args.dropout_iterations):
    #         probabilities = get_probs(model, unlab_loader, stochastic=True)
    #
    #
    #
    #         entropy = - np.multiply(probabilities, np.log(probabilities))
    #         entropy = np.sum(entropy, axis=1)
    #
    #         entropy_sum += entropy
    #         scores_sum += probabilities
    #
    #
    #     avg_scores = np.divide(scores_sum, args.dropout_iterations)
    #     entropy_avg_sc = - np.multiply(avg_scores, np.log(avg_scores))
    #     entropy_avg_sc = np.sum(entropy_avg_sc, axis=1)
    #
    #     avg_entropy = np.divide(entropy_sum, args.dropout_iterations)
    #
    #     bald_score = entropy_avg_sc - avg_entropy
    #
    #     # partial sort
    #     argsorted_bald = np.argpartition(-bald_score, sample_size)
    #     # get the indices
    #     sample_indices = argsorted_bald[:sample_size]
    #     sample_rows = dropout_pool[sample_indices]
    #
    #     return sample_rows


def bound_max_loss(energy, bound):
    """
    return the loss value of max(0, \mathcal{F}(x) - \Delta )
    """
    energy_minus_bound = energy - bound
    energy_minus_bound = torch.unsqueeze(energy_minus_bound, dim=1)
    zeros = torch.zeros_like(energy_minus_bound)
    for_select = torch.cat((energy_minus_bound, zeros), dim=1)
    selected = torch.max(for_select, dim=1).values

    return selected.mean()


class FreeEnergyAlignmentLoss(nn.Module):
    """
    free energy alignment loss
    """

    def __init__(self, cfg):
        super(FreeEnergyAlignmentLoss, self).__init__()
        assert cfg.TRAINER.ENERGY_BETA > 0, "beta for energy calculate must be larger than 0"
        self.beta = cfg.TRAINER.ENERGY_BETA

        self.type = cfg.TRAINER.ENERGY_ALIGN_TYPE

        if self.type == 'l1':
            self.loss = nn.L1Loss()
        elif self.type == 'mse':
            self.loss = nn.MSELoss()
        elif self.type == 'max':
            self.loss = bound_max_loss

    def forward(self, inputs, bound):
        mul_neg_beta = -1.0 * self.beta * inputs
        log_sum_exp = torch.logsumexp(mul_neg_beta, dim=1)
        free_energies = -1.0 * log_sum_exp / self.beta

        bound = torch.ones_like(free_energies) * bound
        loss = self.loss(free_energies, bound)

        return loss



class NLLLoss(nn.Module):
    """
    NLL loss for energy based model
    """

    def __init__(self, cfg):
        super(NLLLoss, self).__init__()
        assert cfg.TRAINER.ENERGY_BETA > 0, "beta for energy calculate must be larger than 0"
        self.beta = cfg.TRAINER.ENERGY_BETA

    def forward(self, inputs, targets):
        indices = torch.unsqueeze(targets, dim=1)
        energy_c = torch.gather(inputs, dim=1, index=indices)

        all_energy = -1.0 * self.beta * inputs
        free_energy = -1.0 * torch.logsumexp(all_energy, dim=1, keepdim=True) / self.beta

        nLL = energy_c - free_energy

        return nLL.mean()
