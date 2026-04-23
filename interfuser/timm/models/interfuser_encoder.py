import math
import copy
import torch
from torch import nn, Tensor
from functools import partial
import logging
from typing import Optional
from collections import OrderedDict
from .registry import register_model
from .resnet import resnet18d, resnet26d, resnet50d, resnet101d
from .layers import to_2tuple

_logger = logging.getLogger(__name__)


class HybridEmbed(nn.Module):
    def __init__(
        self,
        backbone,
        img_size=224,
        patch_size=1,
        feature_size=None,
        in_chans=3,
        embed_dim=768,
    ):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, "feature_info"):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features

        self.proj = nn.Conv2d(feature_dim, embed_dim, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x)
        global_x = torch.mean(x, [2, 3], keepdim=False)[:, :, None]
        return x, global_x


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self, num_pos_feats=64, temperature=10000, normalize=False, scale=None
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        x = tensor
        bs, _, h, w = x.shape
        not_mask = torch.ones((bs, h, w), device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=nn.ReLU(),
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_attn_mask(mask_type):
    mask = torch.ones((151, 151), dtype=torch.bool).cuda()
    if mask_type == "seperate_all":
        mask[:50, :50] = False
        mask[50:67, 50:67] = False
        mask[67:84, 67:84] = False
        mask[84:101, 84:101] = False
        mask[101:151, 101:151] = False
    elif mask_type == "seperate_view":
        mask[:50, :50] = False
        mask[50:67, 50:67] = False
        mask[67:84, 67:84] = False
        mask[84:101, 84:101] = False
        mask[101:151, :] = False
        mask[:, 101:151] = False
    return mask


class InterfuserEncoder(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        enc_depth=6,
        dim_feedforward=2048,
        normalize_before=False,
        rgb_backbone_name="r26",
        lidar_backbone_name="r26",
        num_heads=8,
        dropout=0.1,
        direct_concat=False,
        separate_view_attention=False,
        separate_all_attention=False,
        act_layer=None,
        with_lidar=True,
        with_right_left_sensors=True,
        with_center_sensor=True,
        traffic_pred_head_type="det",
        use_different_backbone=False,
        use_view_embed=True,
        use_mmad_pretrain=None,
    ):
        super().__init__()
        self.traffic_pred_head_type = traffic_pred_head_type
        act_layer = act_layer or nn.GELU

        self.with_lidar = with_lidar
        self.with_right_left_sensors = with_right_left_sensors
        self.with_center_sensor = with_center_sensor

        self.direct_concat = direct_concat
        self.separate_view_attention = separate_view_attention
        self.separate_all_attention = separate_all_attention
        self.use_view_embed = use_view_embed

        if self.direct_concat:
            in_chans = in_chans * 4
            self.with_center_sensor = False
            self.with_right_left_sensors = False

        if self.separate_view_attention:
            self.attn_mask = build_attn_mask("seperate_view")
        elif self.separate_all_attention:
            self.attn_mask = build_attn_mask("seperate_all")
        else:
            self.attn_mask = None

        if use_different_backbone:
            if rgb_backbone_name == "r50":
                self.rgb_backbone = resnet50d(
                    pretrained=True,
                    in_chans=in_chans,
                    features_only=True,
                    out_indices=[4],
                )
            elif rgb_backbone_name == "r26":
                self.rgb_backbone = resnet26d(
                    pretrained=True,
                    in_chans=in_chans,
                    features_only=True,
                    out_indices=[4],
                )
            elif rgb_backbone_name == "r18":
                self.rgb_backbone = resnet18d(
                    pretrained=True,
                    in_chans=in_chans,
                    features_only=True,
                    out_indices=[4],
                )
            if lidar_backbone_name == "r50":
                self.lidar_backbone = resnet50d(
                    pretrained=False,
                    in_chans=in_chans,
                    features_only=True,
                    out_indices=[4],
                )
            elif lidar_backbone_name == "r26":
                self.lidar_backbone = resnet26d(
                    pretrained=False,
                    in_chans=in_chans,
                    features_only=True,
                    out_indices=[4],
                )
            elif lidar_backbone_name == "r18":
                self.lidar_backbone = resnet18d(
                    pretrained=False, in_chans=3, features_only=True, out_indices=[4]
                )
            rgb_embed_layer = partial(HybridEmbed, backbone=self.rgb_backbone)
            lidar_embed_layer = partial(HybridEmbed, backbone=self.lidar_backbone)

            if use_mmad_pretrain:
                params = torch.load(use_mmad_pretrain)["state_dict"]
                updated_params = OrderedDict()
                for key in params:
                    if "backbone" in key:
                        updated_params[key.replace("backbone.", "")] = params[key]
                self.rgb_backbone.load_state_dict(updated_params)

            self.rgb_patch_embed = rgb_embed_layer(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
            self.lidar_patch_embed = lidar_embed_layer(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=3,
                embed_dim=embed_dim,
            )
        else:
            if rgb_backbone_name == "r50":
                self.rgb_backbone = resnet50d(
                    pretrained=True, in_chans=3, features_only=True, out_indices=[4]
                )
            elif rgb_backbone_name == "r101":
                self.rgb_backbone = resnet101d(
                    pretrained=True, in_chans=3, features_only=True, out_indices=[4]
                )
            elif rgb_backbone_name == "r26":
                self.rgb_backbone = resnet26d(
                    pretrained=True, in_chans=3, features_only=True, out_indices=[4]
                )
            elif rgb_backbone_name == "r18":
                self.rgb_backbone = resnet18d(
                    pretrained=True, in_chans=3, features_only=True, out_indices=[4]
                )
            embed_layer = partial(HybridEmbed, backbone=self.rgb_backbone)

            self.rgb_patch_embed = embed_layer(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
            self.lidar_patch_embed = embed_layer(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )

        self.global_embed = nn.Parameter(torch.zeros(1, embed_dim, 5))
        self.view_embed = nn.Parameter(torch.zeros(1, embed_dim, 5, 1))

        if self.traffic_pred_head_type == "det":
            self.traffic_pred_head = nn.Sequential(
                *[
                    nn.Linear(embed_dim + 32, 64),
                    nn.ReLU(),
                    nn.Linear(64, 7),
                    nn.Sigmoid(),
                ]
            )
        elif self.traffic_pred_head_type == "seg":
            self.traffic_pred_head = nn.Sequential(
                *[nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()]
            )

        self.position_encoding = PositionEmbeddingSine(embed_dim // 2, normalize=True)

        encoder_layer = TransformerEncoderLayer(
            embed_dim, num_heads, dim_feedforward, dropout, act_layer, normalize_before
        )
        self.encoder = TransformerEncoder(encoder_layer, enc_depth, None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.global_embed)
        nn.init.uniform_(self.view_embed)

    def forward_features(
        self,
        front_image,
        left_image,
        right_image,
        front_center_image,
        lidar,
    ):
        features = []

        # Front view processing
        front_image_token, front_image_token_global = self.rgb_patch_embed(front_image)
        if self.use_view_embed:
            front_image_token = (
                front_image_token
                + self.view_embed[:, :, 0:1, :]
                + self.position_encoding(front_image_token)
            )
        else:
            front_image_token = front_image_token + self.position_encoding(
                front_image_token
            )
        front_image_token = front_image_token.flatten(2).permute(2, 0, 1)
        front_image_token_global = (
            front_image_token_global
            + self.view_embed[:, :, 0, :]
            + self.global_embed[:, :, 0:1]
        )
        front_image_token_global = front_image_token_global.permute(2, 0, 1)
        features.extend([front_image_token, front_image_token_global])

        if self.with_right_left_sensors:
            # Left view processing
            left_image_token, left_image_token_global = self.rgb_patch_embed(left_image)
            if self.use_view_embed:
                left_image_token = (
                    left_image_token
                    + self.view_embed[:, :, 1:2, :]
                    + self.position_encoding(left_image_token)
                )
            else:
                left_image_token = left_image_token + self.position_encoding(
                    left_image_token
                )
            left_image_token = left_image_token.flatten(2).permute(2, 0, 1)
            left_image_token_global = (
                left_image_token_global
                + self.view_embed[:, :, 1, :]
                + self.global_embed[:, :, 1:2]
            )
            left_image_token_global = left_image_token_global.permute(2, 0, 1)

            # Right view processing
            right_image_token, right_image_token_global = self.rgb_patch_embed(
                right_image
            )
            if self.use_view_embed:
                right_image_token = (
                    right_image_token
                    + self.view_embed[:, :, 2:3, :]
                    + self.position_encoding(right_image_token)
                )
            else:
                right_image_token = right_image_token + self.position_encoding(
                    right_image_token
                )
            right_image_token = right_image_token.flatten(2).permute(2, 0, 1)
            right_image_token_global = (
                right_image_token_global
                + self.view_embed[:, :, 2, :]
                + self.global_embed[:, :, 2:3]
            )
            right_image_token_global = right_image_token_global.permute(2, 0, 1)

            features.extend(
                [
                    left_image_token,
                    left_image_token_global,
                    right_image_token,
                    right_image_token_global,
                ]
            )

        if self.with_center_sensor:
            # Front center view processing
            (
                front_center_image_token,
                front_center_image_token_global,
            ) = self.rgb_patch_embed(front_center_image)
            if self.use_view_embed:
                front_center_image_token = (
                    front_center_image_token
                    + self.view_embed[:, :, 3:4, :]
                    + self.position_encoding(front_center_image_token)
                )
            else:
                front_center_image_token = (
                    front_center_image_token
                    + self.position_encoding(front_center_image_token)
                )

            front_center_image_token = front_center_image_token.flatten(2).permute(
                2, 0, 1
            )
            front_center_image_token_global = (
                front_center_image_token_global
                + self.view_embed[:, :, 3, :]
                + self.global_embed[:, :, 3:4]
            )
            front_center_image_token_global = front_center_image_token_global.permute(
                2, 0, 1
            )
            features.extend([front_center_image_token, front_center_image_token_global])

        if self.with_lidar:
            lidar_token, lidar_token_global = self.lidar_patch_embed(lidar)
            if self.use_view_embed:
                lidar_token = (
                    lidar_token
                    + self.view_embed[:, :, 4:5, :]
                    + self.position_encoding(lidar_token)
                )
            else:
                lidar_token = lidar_token + self.position_encoding(lidar_token)
            lidar_token = lidar_token.flatten(2).permute(2, 0, 1)
            lidar_token_global = (
                lidar_token_global
                + self.view_embed[:, :, 4, :]
                + self.global_embed[:, :, 4:5]
            )
            lidar_token_global = lidar_token_global.permute(2, 0, 1)
            features.extend([lidar_token, lidar_token_global])

        features = torch.cat(features, 0)
        return features

    def forward(self, x):
        front_image = x["rgb"]
        left_image = x["rgb_left"]
        right_image = x["rgb_right"]
        front_center_image = x["rgb_center"]
        lidar = x["lidar"]

        if self.direct_concat:
            img_size = front_image.shape[-1]
            left_image = torch.nn.functional.interpolate(
                left_image, size=(img_size, img_size)
            )
            right_image = torch.nn.functional.interpolate(
                right_image, size=(img_size, img_size)
            )
            front_center_image = torch.nn.functional.interpolate(
                front_center_image, size=(img_size, img_size)
            )
            front_image = torch.cat(
                [front_image, left_image, right_image, front_center_image], dim=1
            )
        features = self.forward_features(
            front_image,
            left_image,
            right_image,
            front_center_image,
            lidar,
        )

        memory = self.encoder(features, mask=self.attn_mask)

        # 分类

        # return traffic, waypoints, is_junction, traffic_light_state, stop_sign, traffic_feature


@register_model
def interfuser_encoder(**kwargs):
    model = InterfuserEncoder(
        enc_depth=6,
        embed_dim=256,
        rgb_backbone_name="r50",
        lidar_backbone_name="r18",
        use_different_backbone=True,
    )
    return model
