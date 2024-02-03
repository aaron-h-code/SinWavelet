import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ConvBlock, TriplaneConvs, Convs3DSkipAdd, make_mlp

from wavelet.pytorch_wavelets.dwt.transform2d import DWTForward, DWTInverse

import pdb


def get_network(config, name):
    """get specificed network

    Args:
        config (Config): a config object
        name (str): "G" for generator, "D" for discriminator

    Returns:
        network (nn.Module)
    """
    if name == "G":
        if config.G_struct == "triplane":
            return GrowingGeneratorTriplane(config.G_nc, config.G_layers, config.pool_dim,
                                            config.feat_dim, config.use_norm, config.mlp_dim, config.mlp_layers)
        elif config.G_struct == "conv3d":
            # return GrowingGenerator3D(config.G_nc, config.G_layers, config.use_norm)
            raise NotImplementedError
        else:
            raise NotImplementedError
    elif name == "D":
        return WDiscriminator(config.D_nc, config.D_layers, config.use_norm, n_augs=config.n_augs)
    else:
        raise NotImplementedError


class WDiscriminator(nn.Module):
    def __init__(self, n_channels=32, n_layers=3, use_norm=True, n_augs=None):
        """A 3D convolutional discriminator.
            Each layer's kernel size, stride and padding size are fixed.

        Args:
            n_channels (int, optional): number of channels for each layer. Defaults to 32.
            n_layers (int, optional): number of conv layers. Defaults to 3.
            use_norm (bool, optional): use normalization layer. Defaults to True.
        """
        super(WDiscriminator, self).__init__()
        # ker_size, stride, pad = 3, 2, 1 # hard-coded
        # self.head = ConvBlock(1, n_channels, ker_size, stride, pad, use_norm, sdim='3d')
        #
        # self.body = nn.Sequential()
        # for i in range(n_layers - 2):
        #     ker_size, stride, pad = 3, 1, 0 # hard-coded
        #     block = ConvBlock(n_channels, n_channels, ker_size, stride, pad, use_norm, sdim='3d')
        #     self.body.add_module('block%d' % (i + 1), block)
        #
        # ker_size, stride, pad = 3, 1, 0 # hard-coded
        # self.tail = nn.Conv3d(n_channels, 1, ker_size, stride, pad)

        # DAG
        self.n_augs = n_augs

        self.head_dag = []
        self.body_dag = []
        self.tail_dag = []
        for i in range(self.n_augs):
            ker_size, stride, pad = 3, 2, 1  # hard-coded
            self.head_dag.append(ConvBlock(1, n_channels, ker_size, stride, pad, use_norm, sdim='3d'))

            body = nn.Sequential()
            for i in range(n_layers - 2):
                ker_size, stride, pad = 3, 1, 0  # hard-coded
                block = ConvBlock(n_channels, n_channels, ker_size, stride, pad, use_norm, sdim='3d')
                body.add_module('block%d' % (i + 1), block)
            self.body_dag.append(body)

            ker_size, stride, pad = 3, 1, 0  # hard-coded
            self.tail_dag.append(nn.Conv3d(n_channels, 1, ker_size, stride, pad))

        self.head_dag = nn.ModuleList(self.head_dag)
        self.body_dag = nn.ModuleList(self.body_dag)
        self.tail_dag = nn.ModuleList(self.tail_dag)

    def forward(self, x, dag_idx=0):
        # x = self.head(x)
        # x = self.body(x)
        # x = self.tail(x)

        # dag heads
        x = self.head_dag[dag_idx](x)
        x = self.body_dag[dag_idx](x)
        x = self.tail_dag[dag_idx](x)

        return x


class GrowingGeneratorTriplane(nn.Module):
    def __init__(self, n_channels=32, n_layers=4, pool_dim=8, feat_dim=32, use_norm=True, mlp_dim=32, mlp_layers=0):
        """A multi-scale generator on tri-plane representation.

        Args:
            n_channels (int, optional): number of channels. Defaults to 32.
            n_layers (int, optional): number of conv layers. Defaults to 4.
            pool_dim (int, optional): average pooling dimension at head. Defaults to 8.
            feat_dim (int, optional): tri-plane feature dimension. Defaults to 32.
            use_norm (bool, optional): use normalization layer. Defaults to True.
            mlp_dim (int, optional): mlp hidden layer feature dimension. Defaults to 32.
            mlp_layers (int, optional): number of mlp hidden layers. Defaults to 0.
        """
        super(GrowingGeneratorTriplane, self).__init__()
        self.n_channels = n_channels
        self.n_layers = n_layers
        self.pool_dim = pool_dim
        self.feat_dim = feat_dim
        self.use_norm = use_norm
        self.mlp_dim = mlp_dim
        self.mlp_layers = mlp_layers

        # 1x1 conv
        self.head_conv = TriplaneConvs(self.pool_dim, self.feat_dim, 1, 1, 0, False)
        self.body = nn.ModuleList([])
        self.mlp = make_mlp(self.feat_dim * 3, 1, self.mlp_dim, self.mlp_layers)

        self.dwt_forward = nn.ModuleList([])
        self.dwt_inverse = nn.ModuleList([])

    @property
    def n_scales(self):
        """current number of scales"""
        return len(self.body)

    def init_next_scale(self, level=0, config=None, device=None):
        """initialize next scale, i.e., append a conv block"""
        # out_c_list = [self.n_channels] * (self.n_layers - 1) + [self.feat_dim]
        ker_size, stride, pad = 3, 1, 1  # hard-coded

        out_c_list = [self.n_channels] * (self.n_layers - 1) + [self.feat_dim]
        if level == 0:
            model = TriplaneConvs(self.feat_dim, out_c_list, ker_size, stride, pad, self.use_norm)
        else:
            model = TriplaneConvs(self.feat_dim * 2, out_c_list, ker_size, stride, pad, self.use_norm)
        self.body.append(model)

        if level > 0:
            self.dwt_forward.append(DWTForward(level, config.wavelet_type, device=device))
            self.dwt_inverse.append(DWTInverse(config.wavelet_type, device=device))

    def query(self, tri_feats: list, coords=None):
        """construct output volume through point quries.

        Args:
            tri_feats (list): tri-plane feature maps, [yz_feat, xz_feat, xy_feat]
            coords (tensor, optional): query points of shape (H, W, D, 3). If None, use the size of tri_feats.
        """
        yz_feat, xz_feat, xy_feat = tri_feats
        in_shape = [*xy_feat.shape[-2:], yz_feat.shape[-1]]

        if coords is None:
            yz_feat = yz_feat.permute(0, 2, 3, 1)  # (1, W, D, nf)
            xz_feat = xz_feat.permute(0, 2, 3, 1)  # (1, H, D, nf)
            xy_feat = xy_feat.permute(0, 2, 3, 1)  # (1, H, W, nf)

            # vol_feat = torch.cat([yz_feat.unsqueeze(1).expand(1, *in_shape, -1),
            #                     xz_feat.unsqueeze(2).expand(1, *in_shape, -1),
            #                     xy_feat.unsqueeze(3).expand(1, *in_shape, -1)], dim=-1)

            vol_feat = torch.cat([yz_feat.unsqueeze(1).expand(yz_feat.shape[0], *in_shape, -1),
                                  xz_feat.unsqueeze(2).expand(xz_feat.shape[0], *in_shape, -1),
                                  xy_feat.unsqueeze(3).expand(xy_feat.shape[0], *in_shape, -1)], dim=-1)

            out = self.mlp(vol_feat).permute(0, 4, 1, 2, 3)
        else:
            # FIXME: should assume coords to be (N, 3)
            # coords shape: (H, W, D, 3)
            c_shape = coords.shape[:3]
            coords = coords.view(-1, 3)
            # to save memory
            out = []
            batch_size = 128 ** 3  # FIXME: hard coded, prevent overflow
            N = coords.shape[0]
            for j in range(N // batch_size + 1):
                coords_ = coords[j * batch_size: (j + 1) * batch_size].unsqueeze(0).unsqueeze(0)  # (1, 1, N, 3)

                sample_yz_feat = F.grid_sample(
                    yz_feat, coords_[..., [1, 2]].flip(-1), align_corners=True).permute(0, 2, 3, 1).squeeze(0).squeeze(0)
                sample_xz_feat = F.grid_sample(
                    xz_feat, coords_[..., [0, 2]].flip(-1), align_corners=True).permute(0, 2, 3, 1).squeeze(0).squeeze(0)
                sample_xy_feat = F.grid_sample(
                    xy_feat, coords_[..., [0, 1]].flip(-1), align_corners=True).permute(0, 2, 3, 1).squeeze(0).squeeze(0)

                vol_feat = torch.cat([sample_yz_feat, sample_xz_feat, sample_xy_feat], dim=-1)
                batch_out = self.mlp(vol_feat)
                out.append(batch_out)

            out = torch.cat(out, dim=0).squeeze(-1).view(c_shape).unsqueeze(0).unsqueeze(0)

        out = torch.sigmoid(out)

        return out

    def forward_head(self, init_noise):
        """forward through the projection module at head."""
        ni = init_noise
        # extract triplane features at head
        in_shape = ni.shape[-3:]

        yz_feat = F.adaptive_avg_pool3d(ni, (self.pool_dim, in_shape[1], in_shape[2])).squeeze(1)
        xz_feat = F.adaptive_avg_pool3d(ni, (in_shape[0], self.pool_dim, in_shape[2])).squeeze(1).permute(0, 2, 1, 3)
        xy_feat = F.adaptive_avg_pool3d(ni, (in_shape[0], in_shape[1], self.pool_dim)).squeeze(1).permute(0, 3, 1, 2)

        yz_feat, xz_feat, xy_feat = self.head_conv([yz_feat, xz_feat, xy_feat], add_noise=False, skip_add=False)

        return [yz_feat, xz_feat, xy_feat]

    def _upsample_triplanes(self, tri_feats: list, up_size: list):
        """upsample tri-plane feature maps"""
        tri_feats[0] = F.interpolate(tri_feats[0], size=(up_size[1], up_size[2]), mode='bilinear', align_corners=True)
        tri_feats[1] = F.interpolate(tri_feats[1], size=(up_size[0], up_size[2]), mode='bilinear', align_corners=True)
        tri_feats[2] = F.interpolate(tri_feats[2], size=(up_size[0], up_size[1]), mode='bilinear', align_corners=True)

        return tri_feats

    def draw_feats(self, init_noise: list, real_sizes: list, noises_list: list, mode: str, end_scale: int,
                   scale=None, dwt_forward=None, dwt_inverse=None, timesteps: int = None, time_emb_dim: int = None):
        """draw generated tri-plane feature maps at end_scale. To facilitate training."""

        tri_feats = self.forward_head(init_noise)

        for i in range(end_scale):
            if i > 0:
                tri_feats = self._upsample_triplanes(tri_feats, real_sizes[i])

                tri_feats_coeffs = [self.dwt_forward[i - 1](x) for x in tri_feats]

                tri_feats_low_coeffs = [x[0] for x in tri_feats_coeffs]
                tri_feats_low_coeffs_up = [F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=True)
                                           for x, y in zip(tri_feats, tri_feats_low_coeffs)]

                tri_feats_high_coeffs = [x[1] for x in tri_feats_coeffs]
                tri_feats_high_coeffs_up = [None] * len(tri_feats_high_coeffs)

                # upsample high coeffs
                for k, (x, y) in enumerate(zip(tri_feats, tri_feats_high_coeffs)):
                    y = torch.cat(
                        [F.interpolate(h.view(h.shape[0], -1, *h.shape[3:]), size=x.shape[2:], mode='bilinear',
                                       align_corners=True).view(h.shape[0], -1, 3, *x.shape[2:]) for h in y], dim=0)
                    tri_feats_high_coeffs_up[k] = y

                tri_feats_cat = [None] * len(tri_feats)
                for k, (x, y) in enumerate(zip(tri_feats_low_coeffs_up, tri_feats_high_coeffs_up)):
                    tri_feats_cat[k] = torch.cat(
                        [x.unsqueeze(2).repeat(self.dwt_forward[i - 1].J, 1, 3, 1, 1), y], dim=1)

                tri_feats_cat = [x.transpose(1, 2).contiguous().view(-1, x.shape[1], *x.shape[3:])
                                 for x in tri_feats_cat]

                tri_feats_high_coeffs_next = self.body[i](tri_feats_cat, noises_list[i],
                                                          add_noise=i > 0 and mode != "rec", skip_add=i > 0)

                tri_feats_high_coeffs_next = [x.view(-1, 3, *x.shape[1:]).transpose(1, 2)
                                              for x in tri_feats_high_coeffs_next]

                # downsample high coeffs
                for k, (x, y) in enumerate(zip(tri_feats_high_coeffs_next, tri_feats_high_coeffs)):
                    x = x.split(split_size=y[0].shape[0], dim=0)
                    y = [F.interpolate(hn.contiguous().view(hn.shape[0], -1, *hn.shape[3:]), size=h.shape[3:],
                                       mode='bilinear', align_corners=True).view(hn.shape[0], -1, 3, *h.shape[3:])
                         for hn, h in zip(x, y)]
                    tri_feats_high_coeffs_next[k] = y

                tri_feats = [self.dwt_inverse[i - 1]([x, y]) for x, y in
                             zip(tri_feats_low_coeffs, tri_feats_high_coeffs_next)]

                tri_feats = self._upsample_triplanes(tri_feats, real_sizes[i])  # slightly adjust
            else:
                tri_feats = self.body[i](tri_feats, noises_list[i], add_noise=i > 0 and mode != "rec", skip_add=i > 0)

        return tri_feats

    def decode_feats(self, tri_feats: list, real_sizes: list, noises_list: list, mode: str, start_scale: int,
                     end_scale=-1):
        """pass tri-plane feature maps from start_scale to end_scale, and decode output volume. 
            To facilitate training."""

        if end_scale == -1:
            end_scale = len(self.body)

        assert end_scale - start_scale == 1

        for i in range(end_scale - start_scale):
            ii = start_scale + i

            if ii > 0:
                tri_feats = self._upsample_triplanes(tri_feats, real_sizes[i])

                tri_feats_coeffs = [self.dwt_forward[ii - 1](x) for x in tri_feats]

                tri_feats_low_coeffs = [x[0] for x in tri_feats_coeffs]
                tri_feats_low_coeffs_up = [F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=True)
                                           for x, y in zip(tri_feats, tri_feats_low_coeffs)]

                tri_feats_high_coeffs = [x[1] for x in tri_feats_coeffs]
                tri_feats_high_coeffs_up = [None] * len(tri_feats_high_coeffs)

                # upsample high coeffs
                for k, (x, y) in enumerate(zip(tri_feats, tri_feats_high_coeffs)):
                    y = torch.cat([F.interpolate(h.view(h.shape[0], -1, *h.shape[3:]), size=x.shape[2:], mode='bilinear',
                                    align_corners=True).view(h.shape[0], -1, 3, *x.shape[2:]) for h in y], dim=0)
                    tri_feats_high_coeffs_up[k] = y

                tri_feats_cat = [None] * len(tri_feats)
                for k, (x, y) in enumerate(zip(tri_feats_low_coeffs_up, tri_feats_high_coeffs_up)):
                    tri_feats_cat[k] = torch.cat(
                        [x.unsqueeze(2).repeat(self.dwt_forward[ii - 1].J, 1, 3, 1, 1), y], dim=1)

                tri_feats_cat = [x.transpose(1, 2).contiguous().view(-1, x.shape[1], *x.shape[3:])
                                 for x in tri_feats_cat]

                tri_feats_high_coeffs_next = self.body[ii](tri_feats_cat, noises_list[i],
                                                           add_noise=ii > 0 and mode != "rec", skip_add=ii > 0)

                tri_feats_high_coeffs_next = [x.view(-1, 3, *x.shape[1:]).transpose(1, 2)
                                              for x in tri_feats_high_coeffs_next]

                # downsample high coeffs
                for k, (x, y) in enumerate(zip(tri_feats_high_coeffs_next, tri_feats_high_coeffs)):
                    x = x.split(split_size=y[0].shape[0], dim=0)
                    y = [F.interpolate(hn.contiguous().view(hn.shape[0], -1, *hn.shape[3:]), size=h.shape[3:],
                                       mode='bilinear', align_corners=True).view(hn.shape[0], -1, 3, *h.shape[3:])
                         for hn, h in zip(x, y)]
                    tri_feats_high_coeffs_next[k] = y

                tri_feats = [self.dwt_inverse[ii - 1]([x, y]) for x, y in
                             zip(tri_feats_low_coeffs, tri_feats_high_coeffs_next)]

                tri_feats = self._upsample_triplanes(tri_feats, real_sizes[i])  # slightly adjust

                out = self.query(tri_feats)
            else:
                raise ValueError

        return out

    def forward(self, init_noise: torch.Tensor, real_sizes: list, noises_list: list, mode: str, coords=None,
                return_each=False,
                scale=None, dwt_forward=None, dwt_inverse=None, timesteps=None, time_emb_dim=None):
        """forward through the model

        Args:
            init_noise (torch.Tensor): input 3D noise tensor
            real_sizes (list): list of multi-scale shape sizes
            noises_list (list): list of multi-scale tri-plane noises
            mode (str): "rand" or "rec"
            coords (torch.Tensor, optional): query point coordinates. Defaults to None.
            return_each (bool, optional): return output at each scale. Defaults to False.

        Returns:
            output: 3D shape volume, or a list of 3D shape volume, or feature maps
        """

        tri_feats = self.forward_head(init_noise)

        out_list = []

        for i, block in enumerate(self.body[:len(real_sizes)]):
            if i > 0:
                tri_feats = self._upsample_triplanes(tri_feats, real_sizes[i])

                tri_feats_coeffs = [self.dwt_forward[i - 1](x) for x in tri_feats]

                tri_feats_low_coeffs = [x[0] for x in tri_feats_coeffs]
                tri_feats_low_coeffs_up = [F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=True)
                                           for x, y in zip(tri_feats, tri_feats_low_coeffs)]

                tri_feats_high_coeffs = [x[1] for x in tri_feats_coeffs]
                tri_feats_high_coeffs_up = [None] * len(tri_feats_high_coeffs)

                # upsample high coeffs
                for k, (x, y) in enumerate(zip(tri_feats, tri_feats_high_coeffs)):
                    y = torch.cat([F.interpolate(h.view(h.shape[0], -1, *h.shape[3:]),
                                                 size=x.shape[2:], mode='bilinear',
                                                 align_corners=True).view(h.shape[0], -1, 3, *x.shape[2:]) for h in y],
                                  dim=0)
                    tri_feats_high_coeffs_up[k] = y

                tri_feats_cat = [None] * len(tri_feats)
                for k, (x, y) in enumerate(zip(tri_feats_low_coeffs_up, tri_feats_high_coeffs_up)):
                    tri_feats_cat[k] = torch.cat(
                        [x.unsqueeze(2).repeat(self.dwt_forward[i - 1].J, 1, 3, 1, 1), y], dim=1)

                tri_feats_cat = [x.transpose(1, 2).contiguous().view(-1, x.shape[1], *x.shape[3:])
                                 for x in tri_feats_cat]

                tri_feats_high_coeffs_next = block(tri_feats_cat, noises_list[i],
                                                   add_noise=i > 0 and mode != "rec", skip_add=i > 0)

                tri_feats_high_coeffs_next = [x.view(-1, 3, *x.shape[1:]).transpose(1, 2)
                                              for x in tri_feats_high_coeffs_next]

                # downsample high coeffs
                for k, (x, y) in enumerate(zip(tri_feats_high_coeffs_next, tri_feats_high_coeffs)):
                    x = x.split(split_size=y[0].shape[0], dim=0)
                    y = [F.interpolate(hn.contiguous().view(hn.shape[0], -1, *hn.shape[3:]), size=h.shape[3:],
                                       mode='bilinear', align_corners=True).view(hn.shape[0], -1, 3, *h.shape[3:])
                         for hn, h in zip(x, y)]
                    tri_feats_high_coeffs_next[k] = y

                tri_feats = [self.dwt_inverse[i - 1]([x, y]) for x, y in
                             zip(tri_feats_low_coeffs, tri_feats_high_coeffs_next)]

                tri_feats = self._upsample_triplanes(tri_feats, real_sizes[i])  # slightly adjust

                out = self.query(tri_feats, coords)
            else:
                tri_feats = block(tri_feats, noises_list[i], add_noise=i > 0 and mode != "rec", skip_add=i > 0)
                out = self.query(tri_feats, coords)

            if return_each:
                # out = self.query(tri_feats, coords)
                out_list.append(out)

        if return_each:
            return out_list

        # out = self.query(tri_feats, coords)

        return out
