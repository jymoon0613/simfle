# --------------------------------------------------------
# References:
# DBT: https://github.com/researchmm/DBTNet
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import torch.nn as nn
import torch
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from .models_mae import MaskedAutoencoderViT

class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super(MLPHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

class ChannelGroupingLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_group):
        super(ChannelGroupingLayer, self).__init__()
        self.num_group = num_group
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

        nn.init.constant_(self.convolution[0].weight, 1.0)
        nn.init.constant_(self.convolution[0].bias, 0.1)

        self.denom = 512.0
        self.loss = 0

    def forward(self, x):

        matrix_act = self.convolution(x)

        tmp = matrix_act + 0.001
        b, c, w, h = tmp.shape
        tmp = tmp.view(int((b*c*w*h)/(w*w)), w*w)
        tmp = F.normalize(tmp, p=2)
        tmp = tmp.view(b, c, w*w)
        tmp = tmp.permute(1, 0, 2)
        tmp = tmp.reshape(c, b*w*h)

        co = tmp.mm(tmp.transpose(1,0))
        co = co.view(1, c*c)
        co = co / b

        gt = torch.ones((self.num_group)).cuda()
        gt = gt.diag()
        gt = gt.reshape((1, 1, self.num_group, self.num_group))
        gt = gt.repeat((1, int((c/self.num_group)*(c/self.num_group)), 1, 1))
        gt = F.pixel_shuffle(gt, upscale_factor=int(c/self.num_group))
        gt = gt.reshape((1, c*c))

        loss_single = torch.sum((co-gt)*(co-gt)*0.001, dim=1)
        loss = loss_single.repeat(b)
        loss = loss / ((c/self.denom)*(c/self.denom))

        self.loss = loss

        return matrix_act

class SemanticMaskingLayer(nn.Module):
    def __init__(self, img_size, num_group, patch_size, mask_ratio):
        super(SemanticMaskingLayer, self).__init__()

        self.img_size = img_size
        self.num_group = num_group
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        # ChannelGrouping
        self.channel_grouping = ChannelGroupingLayer(in_channels=2048, out_channels=2048, num_group=self.num_group)

    def forward(self, x, feat):

        # 1. Channel Grouping
        matrix_act = self.channel_grouping(feat.detach())

        # 2. Channel Aggregation
        b, c, w, h = matrix_act.shape

        bias = torch.arange(c/self.num_group, dtype=torch.int64).reshape(-1, 1)

        ind = torch.randint(0, self.num_group, (b,)).reshape(1, -1) * int(c/self.num_group)

        row_indices = ind + bias

        matrix_act_out = matrix_act[torch.arange(b).unsqueeze(-1), row_indices.permute(1, 0)]
        matrix_act_out = torch.sum(matrix_act_out, dim=1, keepdim=True)

        # 3. Upsample & Patch Masking
        N, L, D = x.shape

        matrix_act_out = F.interpolate(matrix_act_out, (self.img_size, self.img_size), mode='bilinear', align_corners=False)

        patches_att = rearrange(matrix_act_out, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1 = self.patch_size, s2 = self.patch_size)

        len_keep = int(L * (1 - self.mask_ratio))

        noise = patches_att.mean(dim = -1)

        ids_shuffle = torch.argsort(noise, dim=1, descending = True)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        mask = torch.gather(mask, dim=1, index=ids_restore)

        g_loss = self.channel_grouping.loss

        return x_masked, mask, ids_restore, g_loss

class FaceMAE(nn.Module):
    def __init__(self, img_size=112, num_group=32, patch_size=16, mask_ratio=0.75):
        super(FaceMAE, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size

        # SemanticMasking
        self.semantic_masking = SemanticMaskingLayer(img_size=img_size, num_group=num_group, patch_size=patch_size, mask_ratio=mask_ratio)

        # Autoencoder
        self.autoencoder = MaskedAutoencoderViT(
            img_size=112, patch_size=16, embed_dim=768, depth=12, num_heads=12,
            decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_pix_loss=True)

    def forward_encoder(self, x, feat):

        # embed patches
        x = self.autoencoder.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.autoencoder.pos_embed[:, 1:, :]

        # Semantic Masking
        x, mask, ids_restore, g_loss = self.semantic_masking(x, feat)

        # append cls token
        cls_token = self.autoencoder.cls_token + self.autoencoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.autoencoder.blocks:
            x = blk(x)
        x = self.autoencoder.norm(x)

        return x, mask, ids_restore, g_loss

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.autoencoder.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.autoencoder.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.autoencoder.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.autoencoder.decoder_blocks:
            x = blk(x)
        x = self.autoencoder.decoder_norm(x)

        # predictor projection
        x = self.autoencoder.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.autoencoder.patchify(imgs)
        if self.autoencoder.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        return loss

    def forward(self, imgs, feat):
        latent, mask, ids_restore, g_loss = self.forward_encoder(imgs, feat)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        gen_loss = self.forward_loss(imgs, pred, mask)

        return gen_loss, g_loss, pred, mask, latent
