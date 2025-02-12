import copy, math, torch, warnings
import numpy as np
from config import device
from torch.distributions.laplace import Laplace
from server.util import warning_format
#warnings.showwarning = warning_show
warnings.formatwarning = warning_format

class SVGD:
    def __init__(self, hyper_post, kernel, optimizer,
                ):
        """
        distribution: GPR_meta_svgd.hyper_post
        clip_norm: bound to be imposed. None if inactive
        note: calculating original norm and clipped norm is inefficient.
        better to use noisy_norm and set record_norms to False.
        """
        self.hyper_post = hyper_post
        self.K = kernel
        self.optim = optimizer
        self.setup_privacy()

    def setup_privacy(self, clip_norm=None, private=False,
                      epsilon=None, delta=None, record_norms=False):
        self.private = private
        self.clip_norm = clip_norm
        self.epsilon = epsilon
        self.delta = delta
        self.record_norms = record_norms
        # Check privacy options
        if not self.private and ((self.epsilon is not None) or (self.delta is not None)):
            print('[Warning] epsilon is not used')
        if self.private and self.epsilon is None:
            print('[Warning] epsilon is not given')
            self.epsilon = 1
        if self.private and self.clip_norm is None:
            print('[Warning] gradient norm clipping is not given')
            self.clip_norm = 0.1
        #if self.private and not self.task_batch_size == 1:
         #   print('[Warning] task batch size set to 1')
          #  self.task_batch_size = 1

        self.original_norm = []
        self.noisy_norm = []
        self.clipped_norm = []


    def phi(self, particles, *data):
        '''
        returns
        *data is train_data_tuples
        '''
        particles = particles.detach().requires_grad_(True)

        # compute the rbf kernel
        K_XY = self.K(particles, particles.detach())
        # d/d X
        dX_K_XY = - torch.autograd.grad(K_XY.sum(), particles)[0]

        # self.hyper_post is RandomGPMeta
        log_prob_hyp_post = self.hyper_post.log_prob(particles, *data)   # of shape [num_particles]
        dX_log_prob_hyp_post = torch.autograd.grad(
            log_prob_hyp_post.sum(), particles
        )[0]    # d/d_particles ln prob(particles | hyper posteroir)

        res = (K_XY.detach().matmul(dX_log_prob_hyp_post) + dX_K_XY) / particles.size(0)
        return res


    def step(self, particles, *data):
        # create a copy
        if self.record_norms:
            particles_copy = particles.detach()
            optim_copy = copy.deepcopy(self.optim)

        # sample noise
        if self.private:
            # (eps, delta)-DP, use Gaussian mechanism
            if self.delta>0:
                noise_std = 2 * math.log(1.25 / self.delta) * self.clip_norm**2 / self.epsilon**2
                noise = torch.randn(particles.shape) * noise_std ** 0.5
            # eps-DP, use Laplace mechanism
                noise = Laplace(
                    loc=torch.zeros(particles.shape),
                    scale=self.clip_norm/self.epsilon*torch.ones(particles.shape)
                ).sample()
            # check shape
            assert noise.shape==particles.shape
        else:
            noise = torch.zeros(particles.shape)
        noise = noise.to(device)

        # calculate noisy grads
        self.optim.zero_grad()
        particles.grad = -self.phi(particles, *data) - torch.mul(particles, noise)
        # instead of defining a loss and minimizing it, gradient of the particles is
        # provided. grad is of size num_particles * num_prior_params and is a 2D tensor

        # clip norm
        if self.clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(particles, self.clip_norm)
        # record norm
        self.noisy_norm.append(torch.norm(particles.grad.detach(), 2))

        # take a step
        self.optim.step()

        # calculate norm without noise
        if self.record_norms:
            optim_copy.zero_grad()
            particles_copy.grad = -self.phi(particles_copy, *data)
            self.original_norm.append(torch.norm(particles_copy.grad.detach(), 2))
            if self.clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(particles_copy, self.clip_norm)
            self.clipped_norm.append(torch.norm(particles_copy.grad.detach(), 2))


class RBF_Kernel(torch.nn.Module):
    """
      RBF kernel

      :math:`K(x, y) = exp(||x-v||^2 / (2h))

    """

    def __init__(self, bandwidth=None):
        super().__init__()
        self.bandwidth = bandwidth

    def _bandwidth(self, norm_sq):
        # Apply the median heuristic (PyTorch does not give true median)
        if self.bandwidth is None:
            np_dnorm2 = norm_sq.detach().cpu().numpy()
            h = np.median(np_dnorm2) / (2 * np.log(np_dnorm2.shape[0] + 1))
            return np.sqrt(h).item()
        else:
            return self.bandwidth

    def forward(self, X, Y):
        dnorm2 = norm_sq(X, Y)
        bandwidth = self._bandwidth(dnorm2)
        gamma = 1.0 / (1e-8 + 2 * bandwidth ** 2)
        K_XY = (-gamma * dnorm2).exp()

        return K_XY


class IMQSteinKernel(torch.nn.Module):
    r"""
    IMQ (inverse multi-quadratic) kernel

    :math:`K(x, y) = (\alpha + ||x-y||^2/h)^{\beta}`

    """

    def __init__(self, alpha=0.5, beta=-0.5, bandwidth=None):
        super(IMQSteinKernel, self).__init__()
        assert alpha > 0.0, "alpha must be positive."
        assert beta < 0.0, "beta must be negative."
        self.alpha = alpha
        self.beta = beta
        self.bandwidth = bandwidth

    def _bandwidth(self, norm_sq):
        """
        Compute the bandwidth along each dimension using the median pairwise squared distance between particles.
        """
        if self.bandwidth is None:
            num_particles = norm_sq.size(0)
            index = torch.arange(num_particles)
            norm_sq = norm_sq[index > index.unsqueeze(-1), ...]
            median = norm_sq.median(dim=0)[0]
            assert median.shape == norm_sq.shape[-1:]
            return median / math.log(num_particles + 1)
        else:
            return self.bandwidth

    def forward(self, X, Y):
        norm_sq = (X.unsqueeze(0) - Y.unsqueeze(1))**2  # N N D
        assert norm_sq.dim() == 3
        bandwidth = self._bandwidth(norm_sq)  # D
        base_term = self.alpha + torch.sum(norm_sq / bandwidth, dim=-1)
        log_kernel = self.beta * torch.log(base_term)  # N N D
        return log_kernel.exp()

""" Helpers """

def norm_sq(X, Y):
    XX = X.matmul(X.t())
    XY = X.matmul(Y.t())
    YY = Y.matmul(Y.t())
    return -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

