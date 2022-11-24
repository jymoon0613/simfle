import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from .layer import MLPHead, FaceMAE

@torch.no_grad()
def _update_target_network_parameters(online_network, target_network, m = 0.99):
    for param_q, param_k in zip(online_network.parameters(), target_network.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1. - m)

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()

        resnet = models.resnet50(pretrained=False)

        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-2])

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.projection = MLPHead(in_channels=resnet.fc.in_features, mlp_hidden_size=4096, projection_size=256)

    def forward(self, x):

        x = self.encoder(x)
        y = self.avgpool(x)
        y = y.view(y.size(0), -1)
        z = self.projection(y)

        return z, y, x

class SimFLE(nn.Module):
    def __init__(self, n_groups=32, mask_ratio=0.75):
        super(SimFLE, self).__init__()

        self.online_network = ResNet50()

        self.target_network = ResNet50()

        self.facemae = FaceMAE(img_size=112, in_channels=2048, out_channels=2048, patch_size=16, n_groups=n_groups, mask_ratio=mask_ratio)

        self.predictor = MLPHead(in_channels=self.online_network.projection.net[-1].out_features,
                            mlp_hidden_size=4096, projection_size=256)

        self.distil_GFL = MLPHead(in_channels=self.online_network.projection.net[0].in_features,
                            mlp_hidden_size=4096, projection_size=256)

        self.distil_FFL = MLPHead(in_channels=self.facemae.autoencoder.patch_embed.proj.out_channels,
                            mlp_hidden_size=4096, projection_size=256)

        self.initializes_target_network()

    def initializes_target_network(self):
    # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _update_target_network_parameters(self):
        _update_target_network_parameters(self.online_network, self.target_network, m = 0.99)

    def forward(self, origin, inps):

        g_loss = None

        x1, x2 = inps

        p1, p1_kd, feat1 = self.online_network(x1)
        p2, p2_kd, feat2 = self.online_network(x2)

        p1 = self.predictor(p1)
        p2 = self.predictor(p2)

        p1_kd = self.distil_GFL(p1_kd)
        p2_kd = self.distil_GFL(p2_kd)

        with torch.no_grad():

            q2, _, _ = self.target_network(x1)
            q1, _, _ = self.target_network(x2)

            _, _, feat = self.online_network(origin)

        gen_loss, g_loss, pred, mask, latent = self.facemae(origin, feat)

        latent = latent[:, 1:, :].mean(dim=1)

        part_kd = self.distil_FFL(latent)

        return g_loss, gen_loss, p1, p2, q1, q2, p1_kd, p2_kd, part_kd, pred, mask
