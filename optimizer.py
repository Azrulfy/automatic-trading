"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import torch

class GlobalAdam(torch.optim.Adam):
    def __init__(self, params, lr, weight_decay):
        super(GlobalAdam, self).__init__(params, lr=lr, weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
