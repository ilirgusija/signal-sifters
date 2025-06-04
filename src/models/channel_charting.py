import torch
import torch.nn as nn
import torch.nn.functional as F


class CSIProviderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.csi = None

    def set_csi(self, csi):
        # csi: torch.Tensor, expected to be complex dtype
        self.csi = csi

    def forward(self, index):
        # index: tensor of indices to gather
        csi_cplx = self.csi[index]
        # Stack real and imag as last dimension
        return torch.stack([csi_cplx.real, csi_cplx.imag], dim=-1)


class FeatureEngineeringLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, csi):
        # csi: (..., 2), where last dim is [real, imag]
        csi = torch.complex(csi[..., 0], csi[..., 1])

        # csi shape: (d, b, r, m, t) or similar
        csi_sum_by_array = csi.sum(dim=(2, 3))  # sum over r, m

        # sample_autocorrelations: (d, b, r, s, m, n, t)
        if csi.ndim == 6: # if has 
            csi = csi.squeeze(0)  # Now shape is [4651, 4, 2, 4, 64]
        sample_autocorrelations = torch.einsum(
            "dbrmt,dbsnt->dbrsmnt", csi, csi.conj()
        )
        # array_sample_autocorrelations: (d, a, b, t)
        if csi_sum_by_array.ndim == 4: # if has 
            csi_sum_by_array = csi_sum_by_array.squeeze(0)  # Now shape is [4651, 4, 64]
        array_sample_autocorrelations = torch.einsum(
            "dbt,dat->dabt", csi_sum_by_array, csi_sum_by_array.conj()
        )

        # Simple trick to make training converge to better to global optimum:
        # Also provided weighted version of horizontal sample autocorrelations (within same row)
        # It is reasonable to assume that horizontal (azimuth) information is more meaningful anyway since
        # most UEs will be somewhere on the surface.
        horiz1 = sample_autocorrelations[:, :, 0, 0, :, :, :] * 4
        horiz2 = sample_autocorrelations[:, :, 1, 1, :, :, :] * 4

        # Flatten all but batch dimension
        def flatten_except_first(x):
            return x.reshape(x.shape[0], -1)

        sample_autocorrelations_flat = flatten_except_first(
            sample_autocorrelations)
        array_sample_autocorrelations_flat = flatten_except_first(
            array_sample_autocorrelations)
        horiz1_flat = flatten_except_first(horiz1)
        horiz2_flat = flatten_except_first(horiz2)

        # Concatenate real and imag parts
        feature_input = torch.cat([
            sample_autocorrelations_flat.real,
            sample_autocorrelations_flat.imag,
            array_sample_autocorrelations_flat.real,
            array_sample_autocorrelations_flat.imag,
            horiz1_flat.real,
            horiz1_flat.imag,
            horiz2_flat.real,
            horiz2_flat.imag
        ], dim=-1)

        return feature_input


class ChannelChart(nn.Module):
    def __init__(self,
                 csi_time_domain,
                 use_csi_provider=True):
        super().__init__()
        self.csi_time_domain = csi_time_domain
        self.use_csi_provider = use_csi_provider

        self.feature_engineering = FeatureEngineeringLayer()
        dummy_input = torch.zeros(
            (1,) + csi_time_domain.shape[1:] + (2,), dtype=torch.float32)
        feat_dim = self.feature_engineering(dummy_input).shape[-1]
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 1024), nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256), nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128), nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64), nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 2),
        )
        if self.use_csi_provider:
            self.csi_provider = CSIProviderLayer()
            self.csi_provider.set_csi(csi_time_domain.detach().clone())
        else:
            self.csi_provider = None

    def forward(self, csi):
        if self.use_csi_provider:
            csi = self.csi_provider(csi)
        features = self.feature_engineering(csi)
        return self.mlp(features)

    def predict(self, csi_time_domain):
        self.eval()
        with torch.no_grad():
            csi_time_domain_tensor = torch.tensor(
                csi_time_domain, dtype=torch.complex64)
            csi_time_domain_tensor_re_im = torch.stack(
                [csi_time_domain_tensor.real, csi_time_domain_tensor.imag], dim=-1)
            features = self.feature_engineering(csi_time_domain_tensor_re_im)
            return self.mlp(features)
