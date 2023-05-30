import time
import gc
import logging
from collections import OrderedDict
from collections.abc import Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

loss = nn.CrossEntropyLoss()

class Attack:
    def __init__(self):
        pass
    
    def l2(self, x):
        # return torch.sqrt(torch.sum(torch.square(x), (1,2,3))) 
        batch_size = x.shape[0]
        grad_norms = torch.norm(x.reshape(batch_size, -1), p=2, dim=1)
        return grad_norms.reshape(batch_size, 1, 1, 1)

    def l2_pgd_attack(self, model, images, labels, eps, alpha=10/255, iters=5):   
        x0 = images.clone().detach()
        xt = x0.clone().detach()    

        for i in range(iters) :        
            xt.requires_grad = True
            outputs = model(xt)
            
            # getting the gradient grad(loss)
            model.zero_grad()
            cost = loss(outputs, labels)
            cost.backward()

            # apply the gradient to our current point
            grads = xt.grad
            x_next = xt.detach() + grads / self.l2(grads) * alpha
            
            # project the current point on the l2(x0, epsilon) ball :)
            delta = x_next - x0
            sigma = self.l2(delta)
            x_next = x_next + (delta / sigma) * eps
            
            # ready for the next step
            xt = x_next.clamp(min=0, max=1).detach()

        return xt
