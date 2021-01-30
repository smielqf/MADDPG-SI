import numpy as np
import torch
import torch.nn.functional as F
from multiagent.multi_discrete import MultiDiscrete


class Pd(object):
    """
    A particular probability distribution
    """

    def flatparam(self):
        raise NotImplementedError

    def mode(self):
        raise NotImplementedError

    def logp(self, x):
        raise NotImplementedError

    def kl(self, other):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError


class PdType(object):
    """
    Parametrized family of probability distributions
    """

    def pdclass(self):
        raise NotImplementedError

    def pdfromflat(self, flat, device='cpu'):
        return self.pdclass()(flat, device)

    def param_shape(self):
        raise NotImplementedError

    def sample_shape(self):
        raise NotImplementedError

    def sample_dtype(self):
        raise NotImplementedError


class SoftCategoricalPdType(PdType):
    def __init__(self, ncat):
        self.ncat = ncat

    def pdclass(self):
        return SoftCategoricalPd

    def param_shape(self):
        return [self.ncat]

    def sample_shape(self):
        return [self.ncat]

    def sample_dtype(self):
        return torch.float32


class SoftCategoricalPd(Pd):
    def __init__(self, logits, device='cpu'):
        self.logits = logits
        self.device = device

    def flatparam(self):
        return self.logits

    def mode(self):
        return F.softmax(self.logits, dim=-1)

    def logp(self, x):
        """ Calculate log probability of x using negative cross entropy. Since cross entropy ranges from 0 to infinity, it can be minus to range from negative infinity to 0, which matches the range of log probability. 
        
        Parameters
        ----------
        x : Tensor(float), [batch_size, num_action]
            action value.
        
        Returns
        -------
        Tensor(float), [batch_size, ]
            log probability of x.
        """
        a0 = self.logits - torch.max(self.logits, dim=1, keepdim=True).values
        ea0 = torch.exp(a0)
        z0 = torch.sum(ea0, dim=1, keepdim=True)
        p0 = ea0 / z0
        log_p0 = torch.log(p0)
        return torch.sum(x * log_p0, dim=1)

    def kl(self, other):
        a0 = self.logits - torch.max(self.logits, dim=1, keepdim=True).values
        a1 = other.logits - torch.max(other.logits, dim=1, keepdim=True).values
        ea0 = torch.exp(a0)
        ea1 = torch.exp(a1)
        z0 = torch.sum(ea0, dim=1, keepdim=True)
        z1 =torch.sum(ea1, dim=1, keepdim=True)
        p0 = ea0 / z0
        return torch.sum(p0 * (a0 - torch.log(z0) - a1 + torch.log(z1)), dim=1)

    def entropy(self):
        a0 = self.logits - torch.max(self.logits, dim=1, keepdim=True).values
        ea0 = torch.exp(a0)
        z0 = torch.sum(ea0, dim=1, keepdim=True)
        p0 = ea0 / z0
        return torch.sum(p0 * (torch.log(z0) - a0), dim=1)

    def sample(self):
        u = torch.empty(self.logits.size())
        u = torch.nn.init.uniform_(u).to(self.device)
        return torch.softmax(self.logits - torch.log(-torch.log(u)), dim=-1)

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)


def make_pdtype(ac_space):
    from gym import spaces
    if isinstance(ac_space, spaces.Discrete):
        return SoftCategoricalPdType(ac_space.n)
    else:
        raise NotImplementedError


# def shape_el(v, i):
#     maybe = v.get_shape()[i]
#     if maybe is not None:
#         return maybe
#     else:
#         return tf.shape(v)[i]
