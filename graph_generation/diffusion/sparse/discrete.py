import numpy as np
import torch as th
from torch.nn import Module
from torch.nn import functional as F
from torch_geometric.utils import sort_edge_index
from torch_scatter import scatter


class DiscreteGraphDiffusionModel(Module):
    """Preconditioning for discrete diffusion with optional self-conditioning"""

    def __init__(
        self, self_conditioning, num_node_categories, num_hyperedge_categories, num_incidence_categories, num_steps
    ):
        super().__init__()
        self.self_conditioning = self_conditioning
        self.num_node_categories = num_node_categories
        self.num_hyperedge_categories = num_hyperedge_categories
        self.num_incidence_categories = num_incidence_categories
        self.num_steps = num_steps

    def forward(
        self,
        incidence_index,
        batch,
        node_type,
        node_attr,
        hyperedge_attr,
        incidence_attr,
        t,
        model,
        model_kwargs,
        node_attr_self_cond=None,
        hyperedge_attr_self_cond=None,
        incidence_attr_self_cond=None,
    ):
        # embed node, hyperedge and incidence attributes
        node_attr_in = F.one_hot(node_attr, self.num_node_categories).float() * 2 - 1
        hyperedge_attr_in = F.one_hot(hyperedge_attr, self.num_hyperedge_categories).float() * 2 - 1
        incidence_attr_in = F.one_hot(incidence_attr, self.num_incidence_categories).float() * 2 - 1

        # self-conditioning
        if self.self_conditioning:
            if not model.training:
                assert node_attr_self_cond is not None
                assert hyperedge_attr_self_cond is not None
                assert incidence_attr_self_cond is not None
                node_attr_self_cond = th.softmax(node_attr_self_cond, dim=-1)
                hyperedge_attr_self_cond = th.softmax(hyperedge_attr_self_cond, dim=-1)
                incidence_attr_self_cond = th.softmax(incidence_attr_self_cond, dim=-1)
            elif np.random.rand() < 0.5:
                # sample from next time step
                t_next = th.clamp(t + 1, max=self.num_steps)

                # compute self-conditioning
                with th.no_grad():
                    node_attr_self_cond, hyperedge_attr_self_cond, incidence_attr_self_cond = model(
                        incidence_index=incidence_index,
                        batch=batch,
                        node_type=node_type,
                        node_attr=th.cat(
                            [node_attr_in, th.zeros_like(node_attr_in)], dim=-1
                        ),
                        hyperedge_attr=th.cat(
                            [hyperedge_attr_in, th.zeros_like(hyperedge_attr_in)], dim=-1
                        ),
                        incidence_attr=th.cat(
                            [incidence_attr_in, th.zeros_like(incidence_attr_in)], dim=-1
                        ),
                        **dict(
                            model_kwargs, noise_cond=t_next.float() / self.num_steps
                        ),
                    )
                    node_attr_self_cond = (
                        node_attr_self_cond.detach()
                        + F.one_hot(node_attr, self.num_node_categories).float()
                    )
                    hyperedge_attr_self_cond = (
                        hyperedge_attr_self_cond.detach()
                        + F.one_hot(hyperedge_attr, self.num_hyperedge_categories).float()
                    )
                    incidence_attr_self_cond = (
                        incidence_attr_self_cond.detach()
                        + F.one_hot(incidence_attr, self.num_incidence_categories).float()
                    )
                    node_attr_self_cond = th.softmax(node_attr_self_cond, dim=-1)
                    hyperedge_attr_self_cond = th.softmax(hyperedge_attr_self_cond, dim=-1)
                    incidence_attr_self_cond = th.softmax(incidence_attr_self_cond, dim=-1)
            else:
                node_attr_self_cond = th.zeros_like(node_attr_in)
                hyperedge_attr_self_cond = th.zeros_like(hyperedge_attr_in)
                incidence_attr_self_cond = th.zeros_like(incidence_attr_in)

            # concatenate with input
            node_attr_in = th.cat([node_attr_in, node_attr_self_cond], dim=-1)
            hyperedge_attr_in = th.cat([hyperedge_attr_in, hyperedge_attr_self_cond], dim=-1)
            incidence_attr_in = th.cat([incidence_attr_in, incidence_attr_self_cond], dim=-1)

        # predict node and incidence attributes
        node_pred, hyperedge_pred, incidence_pred = model(
            incidence_index=incidence_index,
            batch=batch,
            node_type=node_type,
            node_attr=node_attr_in,
            hyperedge_attr=hyperedge_attr_in,
            incidence_attr=incidence_attr_in,
            **dict(model_kwargs, noise_cond=t.float() / self.num_steps),
        )
        node_pred = node_pred + F.one_hot(node_attr, self.num_node_categories).float()
        hyperedge_pred = hyperedge_pred + F.one_hot(hyperedge_attr, self.num_hyperedge_categories).float()
        incidence_pred = incidence_pred + F.one_hot(incidence_attr, self.num_incidence_categories).float()

        return node_pred, hyperedge_pred, incidence_pred


class DiscreteGraphDiffusion:
    def __init__(self, self_conditioning, num_steps):
        super().__init__()

        self.model_wrapper = DiscreteGraphDiffusionModel(
            self_conditioning, 2, 3, 2, num_steps
        )
        self.num_steps = num_steps
        self.node_diffusion = CategoricalDiffusion(2, num_steps)
        self.hyperedge_diffusion = CategoricalDiffusion(3, num_steps)
        self.incidence_diffusion = CategoricalDiffusion(2, num_steps)

    @property
    def device(self):
        assert hasattr(self, "_device")
        return self._device

    def to(self, device):
        self._device = device
        self.model_wrapper.to(device)
        self.node_diffusion.to(device)
        self.hyperedge_diffusion.to(device)
        self.incidence_diffusion.to(device)
        return self

    @th.no_grad()
    def sample(self, incidence_index, node_type, batch, model, model_kwargs):
        """Generate samples using the model.

        Iteratively sample from p(x_{t-1} | x_t) for t = T-1, ..., 0, starting from x_T ~ p(x_T).
        """
        num_nodes = scatter(node_type, batch)
        
        # sample from p(x_T)
        node_attr_t = sample_categorical(self.node_diffusion.qT, num_nodes.sum())
        hyperedge_attr_t = sample_categorical(self.hyperedge_diffusion.qT, batch.size(0) - num_nodes.sum())
        incidence_triu_mask = incidence_index[0] < incidence_index[1]
        incidence_triu_index = incidence_index[:, incidence_triu_mask]
        incidence_attr_t = sample_categorical(
            self.incidence_diffusion.qT, incidence_triu_index.size(1)
        )
        incidence_attr_t = to_undirected(incidence_triu_index, incidence_attr_t)[1]

        node_pred = th.zeros(num_nodes.sum(), 2, device=self.device)
        hyperedge_pred = th.zeros(batch.size(0) - num_nodes.sum(), 3, device=self.device)
        incidence_pred = th.zeros(incidence_index.size(1), 2, device=self.device)

        bs = batch.max().item() + 1
        # sample from p(x_{t-1} | x_t) for t = T-1, ..., 0
        for timestep in reversed(range(self.num_steps)):
            t = timestep * th.ones(bs, dtype=th.long, device=self.device)

            # predict node and incidence attributes
            node_pred, hyperedge_pred, incidence_pred = self.model_wrapper(
                incidence_index=incidence_index,
                batch=batch,
                node_type=node_type,
                node_attr=node_attr_t,
                hyperedge_attr=hyperedge_attr_t,
                incidence_attr=incidence_attr_t,
                t=t,
                model=model,
                model_kwargs=model_kwargs,
                node_attr_self_cond=node_pred,
                hyperedge_attr_self_cond=hyperedge_pred,
                incidence_attr_self_cond=incidence_pred,
            )

            # sample ancestor node and incidence attributes
            node_attr_t = self.node_diffusion.q_reverse_sample(
                node_attr_t, node_pred, batch[node_type==1], t
            )
            
            hyperedge_attr_t = self.hyperedge_diffusion.q_reverse_sample(
                hyperedge_attr_t, hyperedge_pred, batch[node_type==0], t
            )

            incidence_attr_t = self.incidence_diffusion.q_reverse_sample(
                incidence_attr_t[incidence_triu_mask],
                incidence_pred[incidence_triu_mask],
                batch[incidence_triu_index[0]],
                t,
            )
            incidence_attr_t = to_undirected(incidence_triu_index, incidence_attr_t)[1]

        # return probabilities for positive class
        node_out = F.softmax(node_pred, dim=-1)[:, 1]
        hyperedge_out = F.softmax(hyperedge_pred, dim=-1)[:, 1]
        incidence_out = F.softmax(incidence_pred, dim=-1)[:, 1]
        return node_out, hyperedge_out, incidence_out

    def get_loss(self, incidence_index, batch, node_type, node_attr, hyperedge_attr, incidence_attr, model, model_kwargs):
        """Compute loss to train the model.

        Sample x_pred ~ p(x_t, t), where t ~ U(0, T-1) and x_t ~ q(x_t | x),
        and compute the cross entropy loss between x_pred and x.
        """
        # sample noisy augmented node and incidence attributes
        t = th.randint(0, self.num_steps, (batch.max().item() + 1,), device=self.device)
        node_attr_t = self.node_diffusion.q_sample(node_attr, batch[node_type==1], t)
        hyperedge_attr_t = self.hyperedge_diffusion.q_sample(hyperedge_attr, batch[node_type==0], t)
        incidence_triu_mask = incidence_index[0] < incidence_index[1]
        incidence_triu_index = incidence_index[:, incidence_triu_mask]
        incidence_triu_attr = incidence_attr[incidence_triu_mask]
        incidence_triu_batch = batch[incidence_index[0, incidence_triu_mask]]
        incidence_triu_attr_t = self.incidence_diffusion.q_sample(
            incidence_triu_attr, incidence_triu_batch, t
        )
        new_incidence_index, incidence_attr_t = to_undirected(incidence_triu_index, incidence_triu_attr_t)
        assert (new_incidence_index == incidence_index).all()

        # predict
        node_pred, hyperedge_pred, incidence_pred = self.model_wrapper(
            incidence_index=incidence_index,
            batch=batch,
            node_type=node_type,
            node_attr=node_attr_t,
            hyperedge_attr=hyperedge_attr_t,
            incidence_attr=incidence_attr_t,
            t=t,
            model=model,
            model_kwargs=model_kwargs,
        )

        # compute loss
        node_loss = self.node_diffusion.get_loss(node_attr, node_pred)
        hyperedge_loss = self.hyperedge_diffusion.get_loss(hyperedge_attr, hyperedge_pred)
        incidence_loss = self.incidence_diffusion.get_loss(incidence_attr, incidence_pred)

        return node_loss, hyperedge_loss, incidence_loss


class CategoricalDiffusion(Module):
    """Discrete diffusion for categorical variables.

    Uses a uniform transition matrix with cosine schedule.
    """

    def __init__(self, num_categories, num_steps):
        super().__init__()

        # stable distribution
        qT = np.ones(num_categories) / num_categories

        # cosine schedule (https://arxiv.org/pdf/2102.05379.pdf)
        steps = np.arange(num_steps + 1, dtype=np.float64) / num_steps
        alphas_bar = np.cos((steps + 0.008) / 1.008 * np.pi / 2)
        betas = np.minimum(1 - alphas_bar[1:] / alphas_bar[:-1], 0.999)

        # single step transition matrices
        Qs = (1 - betas)[:, None, None] * np.eye(num_categories) + betas[
            :, None, None
        ] * qT[None, :, None]

        # t-step transition matrices
        Qbs = (
            alphas_bar[:, None, None] * np.eye(num_categories)
            + (1 - alphas_bar)[:, None, None] * qT[None, :, None]
        )

        # register buffers
        self.register_buffer("qT", th.tensor(qT, dtype=th.float64), persistent=False)
        self.register_buffer("Qs", th.tensor(Qs, dtype=th.float64), persistent=False)
        self.register_buffer("Qbs", th.tensor(Qbs, dtype=th.float64), persistent=False)

    def q_sample(self, x, batch, t):
        """Sample from q(x_t | x)."""
        x_t_prob = self.Qbs[t[batch], x]  # N, |x_t|
        return sample_categorical(x_t_prob, 1).squeeze(1)

    def q_reverse_sample(self, x_t, pred, batch, t):
        """Sample from q(x_{t-1} | x_t) = Σ_x q(x_{t-1} | x, x_t) q(x | x_t)."""
        # compute probs of  posterior q(x_{t-1} | x, x_t) ∝ q(x_t | x_{t-1}) q(x_{t-1} | x) for all x
        left_term = self.Qs[t[batch], :, x_t]  # N, |x_{t-1}|
        right_term = self.Qbs[t - 1][batch]  # N, |x|, |x_{t-1}|
        posterior_probs = left_term[:, None, :] * right_term  # N, |x|, |x_t_1|
        posterior_probs = posterior_probs / posterior_probs.sum(-1, keepdim=True)

        # sample from ancestral distribution q(x_{t-1} | x_t)
        x_probs = F.softmax(pred, dim=-1)  # N, |x|
        ancestral_probs = (posterior_probs * x_probs[..., None]).sum(1)  # N, |x_{t-1}|
        return sample_categorical(ancestral_probs, 1).squeeze(1)  # N

    def get_loss(self, x, pred):
        return F.cross_entropy(pred, x, reduction="none")


def to_undirected(incidence_index, incidence_attr) -> tuple[th.Tensor, th.Tensor]:
    """Convert directed incidences to undirected incidences."""
    incidence_index = th.cat([incidence_index, incidence_index.flip(0)], dim=1)
    incidence_attr = th.cat([incidence_attr, incidence_attr], dim=0)
    return sort_edge_index(incidence_index, incidence_attr)


def sample_categorical(probs, num_samples):
    """Sample from a categorical distribution."""

    if num_samples == 0:
        return th.tensor([], dtype=th.long, device=probs.device).view(
            probs.shape[:-1] + (0,)
        )
    else:
        return th.multinomial(probs, num_samples, replacement=True)