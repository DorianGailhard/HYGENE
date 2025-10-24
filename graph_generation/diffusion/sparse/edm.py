import numpy as np
import torch as th
from torch.nn import Module
from torch_geometric.utils import sort_edge_index


class EDMModel(Module):
    """Preconditioning model for EDM with optional self-conditioning."""

    def __init__(self, self_conditioning):
        super().__init__()
        self.self_conditioning = self_conditioning

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
        # compute weights
        sigma_data = EDM.sigma_data
        c_in = (1 / (sigma_data**2 + t**2).sqrt())[:, None]
        c_skip = (sigma_data**2 / (t**2 + sigma_data**2))[:, None]
        c_out = (t * sigma_data / (t**2 + sigma_data**2).sqrt())[:, None]

        # compute input
        model_kwargs = dict(model_kwargs, noise_cond=(t.log() / 4.0).float())
        node_attr_in = c_in[batch[node_type == 1]] * node_attr
        hyperedge_attr_in = c_in[batch[node_type == 0]] * hyperedge_attr
        incidence_batch = batch[incidence_index[0]]
        incidence_attr_in = c_in[incidence_batch] * incidence_attr

        # self-conditioning
        if self.self_conditioning:
            if not model.training:
                assert (
                    node_attr_self_cond is not None and hyperedge_attr_self_cond is not None and incidence_attr_self_cond is not None
                )
            elif np.random.rand() < 0.5:
                # compute self-conditioning
                with th.no_grad():
                    node_attr_self_cond, hyperedge_attr_self_cond, incidence_attr_self_cond = model(
                        incidence_index=incidence_index,
                        batch=batch,
                        node_type=node_type,
                        node_attr=th.cat(
                            [node_attr_in, th.zeros_like(node_attr_in)], dim=-1
                        ).float(),
                        hyperedge_attr=th.cat(
                            [hyperedge_attr_in, th.zeros_like(hyperedge_attr_in)], dim=-1
                        ).float(),
                        incidence_attr=th.cat(
                            [incidence_attr_in, th.zeros_like(incidence_attr_in)], dim=-1
                        ).float(),
                        **model_kwargs,
                    )
                    node_attr_self_cond = (
                        c_skip[batch[node_type == 1]] * node_attr + c_out[batch[node_type == 1]] * node_attr_self_cond
                    ).detach()
                    hyperedge_attr_self_cond = (
                        c_skip[batch[node_type == 0]] * hyperedge_attr + c_out[batch[node_type == 0]] * hyperedge_attr_self_cond
                    ).detach()
                    incidence_attr_self_cond = (
                        c_skip[incidence_batch] * incidence_attr
                        + c_out[incidence_batch] * incidence_attr_self_cond
                    ).detach()
            else:
                node_attr_self_cond = th.zeros_like(node_attr)
                hyperedge_attr_self_cond = th.zeros_like(hyperedge_attr)
                incidence_attr_self_cond = th.zeros_like(incidence_attr)

            # scale self-conditioning
            node_attr_self_cond = node_attr_self_cond / sigma_data
            hyperedge_attr_self_cond = hyperedge_attr_self_cond / sigma_data
            incidence_attr_self_cond = incidence_attr_self_cond / sigma_data

            # concatenate with input
            node_attr_in = th.cat([node_attr_in, node_attr_self_cond], dim=-1)
            hyperedge_attr_in = th.cat([hyperedge_attr_in, hyperedge_attr_self_cond], dim=-1)
            incidence_attr_in = th.cat([incidence_attr_in, incidence_attr_self_cond], dim=-1)

        # compute output
        node_attr_pred, hyperedge_attr_pred, incidence_attr_pred = model(
            incidence_index=incidence_index,
            batch=batch,
            node_type=node_type,
            node_attr=node_attr_in.float(),
            hyperedge_attr=hyperedge_attr_in.float(),
            incidence_attr=incidence_attr_in.float(),
            **model_kwargs,
        )
        node_attr_pred = c_skip[batch[node_type == 1]] * node_attr + c_out[batch[node_type == 1]] * node_attr_pred
        hyperedge_attr_pred = c_skip[batch[node_type == 0]] * hyperedge_attr + c_out[batch[node_type == 0]] * hyperedge_attr_pred
        incidence_attr_pred = (
            c_skip[incidence_batch] * incidence_attr + c_out[incidence_batch] * incidence_attr_pred
        )

        return node_attr_pred, hyperedge_attr_pred, incidence_attr_pred


class EDM:
    P_mean = -1.2
    P_std = 1.2
    sigma_data = 0.5
    sigma_min = 0.002
    sigma_max = 80
    rho = 7
    S_min = 0.05
    S_max = 50
    S_noise = 1.003
    S_churn = 40

    def __init__(self, self_conditioning, num_steps):
        self.model_wrapper = EDMModel(self_conditioning)
        self.num_steps = num_steps

    @property
    def device(self):
        assert hasattr(self, "_device")
        return self._device

    def to(self, device):
        self._device = device
        self.model_wrapper.to(device)
        return self

    def get_loss(self, incidence_index, batch, node_type, node_attr, hyperedge_attr, incidence_attr, model, model_kwargs):
        # rescale attributes to {-1, 1}
        node_attr = node_attr.float() * 2 - 1
        hyperedge_attr = hyperedge_attr.float() - 1
        incidence_attr = incidence_attr.float() * 2 - 1
        
        # sample noise level
        num_graphs = batch.max().item() + 1
        rnd_normal = th.randn((num_graphs,), device=self.device)
        t = (rnd_normal * self.P_std + self.P_mean).exp()

        # sample noise
        incidence_batch = batch[incidence_index[0]]

        node_noise = th.randn_like(node_attr) * t[batch[node_type == 1]]
        hyperedge_noise = th.randn_like(hyperedge_attr) * t[batch[node_type == 0]]
        incidence_noise = self.incidence_randn(incidence_index) * t[incidence_batch]

        # make prediction
        node_pred, hyperedge_pred, incidence_pred = self.model_wrapper(
            incidence_index=incidence_index,
            batch=batch,
            node_type=node_type,
            node_attr=(node_attr + node_noise).unsqueeze(1),
            hyperedge_attr=(hyperedge_attr + hyperedge_noise).unsqueeze(1),
            incidence_attr=(incidence_attr + incidence_noise).unsqueeze(1),
            t=t,
            model=model,
            model_kwargs=model_kwargs,
        )
        node_pred = node_pred.float().squeeze(1)
        hyperedge_pred = hyperedge_pred.float().squeeze(1)
        incidence_pred = incidence_pred.float().squeeze(1)

        # compute loss
        weight = (t**2 + self.sigma_data**2) / (t * self.sigma_data) ** 2
        node_loss = weight[batch[node_type == 1]] * (node_pred - node_attr) ** 2
        hyperedge_loss = weight[batch[node_type == 0]] * (hyperedge_pred - hyperedge_attr) ** 2
        incidence_loss = weight[incidence_batch] * (incidence_pred - incidence_attr) ** 2
            
        return node_loss, hyperedge_loss, incidence_loss

    @th.no_grad()
    def sample(self, incidence_index, node_type, batch, model, model_kwargs):
        num_graphs = batch.max().item() + 1

        # time step discretization
        step_indices = th.arange(self.num_steps, dtype=th.float64, device=self.device)
        t_steps = (
            self.sigma_max ** (1 / self.rho)
            + step_indices
            / (self.num_steps - 1)
            * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        ) ** self.rho
        t_steps = th.cat([t_steps, th.zeros_like(t_steps[:1])])  # t_N = 0

        # sample latents
        node_attr_next = th.randn_like(batch[node_type == 1], dtype=th.float64)[:, None] * t_steps[0]
        hyperedge_attr_next = th.randn_like(batch[node_type == 0], dtype=th.float64)[:, None] * t_steps[0]
        incidence_attr_next = (
            self.incidence_randn(incidence_index, dtype=th.float64)[:, None] * t_steps[0]
        )

        node_attr_pred = th.zeros_like(node_attr_next)
        hyperedge_attr_pred = th.zeros_like(hyperedge_attr_next)
        incidence_attr_pred = th.zeros_like(incidence_attr_next)
        # sample loop
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            node_attr_cur = node_attr_next
            hyperedge_attr_cur = hyperedge_attr_next
            incidence_attr_cur = incidence_attr_next

            # increase noise temporarily
            gamma = (
                min(self.S_churn / self.num_steps, np.sqrt(2) - 1)
                if self.S_min <= t_cur <= self.S_max
                else 0
            )
            t_hat = t_cur + gamma * t_cur
            node_attr_hat = node_attr_cur + (
                t_hat**2 - t_cur**2
            ).sqrt() * self.S_noise * th.randn_like(node_attr_cur)
            hyperedge_attr_hat = hyperedge_attr_cur + (
                t_hat**2 - t_cur**2
            ).sqrt() * self.S_noise * th.randn_like(hyperedge_attr_cur)
            incidence_attr_hat = (
                incidence_attr_cur
                + (t_hat**2 - t_cur**2).sqrt()
                * self.S_noise
                * self.incidence_randn(incidence_index, dtype=th.float64)[:, None]
            )

            # Euler step
            node_attr_pred, hyperedge_attr_pred, incidence_attr_pred = self.model_wrapper(
                incidence_index=incidence_index,
                batch=batch,
                node_type=node_type,
                node_attr=node_attr_hat,
                hyperedge_attr=hyperedge_attr_hat,
                incidence_attr=incidence_attr_hat,
                t=t_hat.repeat(num_graphs),
                model=model,
                model_kwargs=model_kwargs,
                node_attr_self_cond=node_attr_pred,
                hyperedge_attr_self_cond=hyperedge_attr_pred,
                incidence_attr_self_cond=incidence_attr_pred,
            )
            node_attr_d = (node_attr_hat - node_attr_pred) / t_hat
            hyperedge_attr_d = (hyperedge_attr_hat - hyperedge_attr_pred) / t_hat
            incidence_attr_d = (incidence_attr_hat - incidence_attr_pred) / t_hat
            node_attr_next = node_attr_hat + (t_next - t_hat) * node_attr_d
            hyperedge_attr_next = hyperedge_attr_hat + (t_next - t_hat) * hyperedge_attr_d
            incidence_attr_next = incidence_attr_hat + (t_next - t_hat) * incidence_attr_d

            # 2nd order correction
            if i < self.num_steps - 1:
                node_attr_pred, hyperedge_attr_pred, incidence_attr_pred = self.model_wrapper(
                    incidence_index=incidence_index,
                    batch=batch,
                    node_type=node_type,
                    node_attr=node_attr_next,
                    hyperedge_attr=hyperedge_attr_next,
                    incidence_attr=incidence_attr_next,
                    t=t_next.repeat(num_graphs),
                    model=model,
                    model_kwargs=model_kwargs,
                    node_attr_self_cond=node_attr_pred,
                    hyperedge_attr_self_cond=hyperedge_attr_pred,
                    incidence_attr_self_cond=incidence_attr_pred,
                )
                node_attr_d_prime = (node_attr_next - node_attr_pred) / t_next
                hyperedge_attr_d_prime = (hyperedge_attr_next - hyperedge_attr_pred) / t_next
                incidence_attr_d_prime = (incidence_attr_next - incidence_attr_pred) / t_next
                node_attr_next = node_attr_hat + (t_next - t_hat) * (
                    0.5 * node_attr_d + 0.5 * node_attr_d_prime
                )
                hyperedge_attr_next = hyperedge_attr_hat + (t_next - t_hat) * (
                    0.5 * hyperedge_attr_d + 0.5 * hyperedge_attr_d_prime
                )
                incidence_attr_next = incidence_attr_hat + (t_next - t_hat) * (
                    0.5 * incidence_attr_d + 0.5 * incidence_attr_d_prime
                )

        # rescale attributes to {0, 1} and {0, 2}
        node_attr_out = (node_attr_next + 1) / 2
        hyperedge_attr_out = hyperedge_attr_next + 1
        incidence_attr_out = (incidence_attr_next + 1) / 2

        return node_attr_out.squeeze(1), hyperedge_attr_out.squeeze(1), incidence_attr_out.squeeze(1)

    @staticmethod
    def incidence_randn(incidence_index, dtype=th.float32) -> th.Tensor:
        """Sample symmetric Gaussian noise for incidences in incidence_index."""
        # sample noise for upper triangle
        incidence_index_u = incidence_index[:, incidence_index[0] <= incidence_index[1]]
        incidence_noise_u = th.randn_like(incidence_index_u[0], dtype=dtype)

        # make symmetric
        new_incidence_index, incidence_noise = sort_edge_index(
            edge_index=th.cat([incidence_index_u, incidence_index_u.flip(0)], dim=1),
            edge_attr=th.cat([incidence_noise_u, incidence_noise_u], dim=0),
        )
        assert (incidence_index == new_incidence_index).all()

        return incidence_noise