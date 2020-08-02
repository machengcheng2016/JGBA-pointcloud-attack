import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import utils.pytorch_utils as pytorch_utils
from sklearn.preprocessing import normalize
import os

clip_min = -1.0
clip_max = 1.0
loss_fn = nn.CrossEntropyLoss()
top_k = 10
num_std = 1.0

nbrs = NearestNeighbors(n_neighbors=top_k+1, algorithm='auto', metric='euclidean', n_jobs=-1)

def remove_outliers_defense(x, top_k=10, num_std=1.0):
    top_k = int(top_k)
    num_std = float(num_std)
    if len(x.shape) == 3:
        x = x[0]

    nbrs.fit(x)
    dists = nbrs.kneighbors(x, n_neighbors=top_k + 1)[0][:, 1:]
    dists = np.mean(dists, axis=1)

    avg = np.mean(dists)
    std = num_std * np.std(dists)

    remove_indices = np.where(dists > (avg + std))[0]
    
    save_indices = np.where(dists <= (avg + std))[0]
    x_remove = np.delete(np.copy(x), remove_indices, axis=0)
    return save_indices, x_remove

def remove_outliers_defense_multi(x, top_k=10, num_stds = [0.5, 0.6, 0.7, 0.8, 0.9]):
    top_k = int(top_k)

    if len(x.shape) == 3:
        x = x[0]

    nbrs.fit(x)
    dists = nbrs.kneighbors(x, n_neighbors=top_k + 1)[0][:, 1:]
    dists = np.mean(dists, axis=1)

    avg = np.mean(dists)
    
    save_indices_candidates = []
    x_remove_candidates = []
    for num_std in num_stds:
        std = num_std * np.std(dists)
        remove_indices = np.where(dists > (avg + std))[0]        
        save_indices = np.where(dists <= (avg + std))[0]
        x_remove = np.delete(np.copy(x), remove_indices, axis=0)
        save_indices_candidates.append(save_indices)
        x_remove_candidates.append(x_remove)
    return save_indices_candidates, x_remove_candidates

def JGBA(model, x, y, params):
    eps = float(params["eps"])
    eps_iter = float(params["eps_iter"])
    n = int(params["n"])

    if len(x.shape) == 3:
        x = x[0]

    x_adv = np.copy(x)
    yvar = pytorch_utils.to_var(torch.LongTensor([y]), cuda=True)

    for i in range(n):
        indices_saved, x_sor = remove_outliers_defense(x_adv, top_k=top_k, num_std=num_std)

        xvar = pytorch_utils.to_var(torch.from_numpy(x_sor[None,:,:]), cuda=True, requires_grad=True)
        outputs = model(xvar)
        loss = loss_fn(outputs, yvar)
        loss.backward()
        grad_np = xvar.grad.detach().cpu().numpy()[0]

        xvar_should = pytorch_utils.to_var(torch.from_numpy(x_adv[None,:,:]), cuda=True, requires_grad=True)
        outputs_should = model(xvar_should)
        loss_should = loss_fn(outputs_should, yvar)
        loss_should.backward()
        grad_1024 = xvar_should.grad.detach().cpu().numpy()[0]

        grad_sor = np.zeros((1024, 3))
        
        for idx, index_saved in enumerate(indices_saved):
            grad_sor[index_saved,:] = grad_np[idx,:]
            
        grad_1024 += grad_sor
        grad_1024 = normalize(grad_1024, axis=1)
        
        perturb = eps_iter * grad_1024
        perturb = np.clip(x_adv + perturb, clip_min, clip_max) - x_adv
        norm = np.linalg.norm(perturb, axis=1)
        factor = np.minimum(eps / (norm + 1e-12), np.ones_like(norm))
        factor = np.tile(factor, (3,1)).transpose()
        perturb *= factor
        x_adv += perturb

    x_perturb = np.copy(x_adv)

    return x_perturb

def JGBA_sw(model, x, y, params):
    eps = float(params["eps"])
    eps_iter = float(params["eps_iter"])
    n = int(params["n"])

    if len(x.shape) == 3:
        x = x[0]

    x_adv = np.copy(x)
    yvar = pytorch_utils.to_var(torch.LongTensor([y]), cuda=True)

    for i in range(n):
        indices_saved_cands, x_sor_cands = remove_outliers_defense_multi(x_adv, top_k=top_k, num_stds=[0.5, 0.6, 0.7, 0.8, 0.9])

        xvar_should = pytorch_utils.to_var(torch.from_numpy(x_adv[None,:,:]), cuda=True, requires_grad=True)
        outputs_should = model(xvar_should)
        loss_should = loss_fn(outputs_should, yvar)
        loss_should.backward()
        grad_1024 = xvar_should.grad.detach().cpu().numpy()[0]
        
        grad_1024_cands = []
        
        for (indices_saved_cand, x_sor_cand) in zip(indices_saved_cands, x_sor_cands):
            xvar = pytorch_utils.to_var(torch.from_numpy(x_sor_cand[None,:,:]), cuda=True, requires_grad=True)
            outputs = model(xvar)
            loss = loss_fn(outputs, yvar)
            loss.backward()
            grad_np = xvar.grad.detach().cpu().numpy()[0]

            grad_1024_cand = np.zeros((1024, 3))

            for idx, index_saved in enumerate(indices_saved_cand):
                grad_1024_cand[index_saved,:] = grad_np[idx,:]
            
            grad_1024_cands.append(grad_1024_cand)
        
        grad_1024_cands_mean = np.mean(np.asarray(grad_1024_cands), axis=0)
            
        grad_1024 += grad_1024_cands_mean
        grad_1024 = normalize(grad_1024, axis=1)
        perturb = eps_iter * grad_1024
        
        perturb = np.clip(x_adv + perturb, clip_min, clip_max) - x_adv
        norm = np.linalg.norm(perturb, axis=1)
        factor = np.minimum(eps / (norm + 1e-12), np.ones_like(norm))
        factor = np.tile(factor, (3,1)).transpose()
        perturb *= factor
        x_adv += perturb

    x_perturb = np.copy(x_adv)

    return x_perturb
