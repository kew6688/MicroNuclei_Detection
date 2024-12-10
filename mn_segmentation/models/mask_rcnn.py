"""
Custom mask rcnn based on PyTorch MaskRCNN class. The class is copied to prevent future change in library. 
This code is adapted from PyTorch MaskRCNN implementation.
"""

from collections import OrderedDict
from typing import Any, Callable, Optional

from torch import nn
import torchvision
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import misc as misc_nn_ops
from torchvision.transforms._presets import ObjectDetection
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _COCO_CATEGORIES
from torchvision.models._utils import _ovewrite_value_param, handle_legacy_interface
from torchvision.models.resnet import resnet50, ResNet50_Weights
from torchvision.models.resnet import resnet50, ResNet50_Weights
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.models.detection.faster_rcnn import _default_anchorgen, FasterRCNN, FastRCNNConvFCHead, RPNHead, FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNN

__all__ = [
    "MaskRCNN",
    "MaskRCNN_ResNet50_FPN_Weights",
    "MaskRCNN_ResNet50_FPN_V2_Weights",
    "maskrcnn_resnet50_fpn",
    "maskrcnn_resnet50_fpn_v2",
]


class MaskRCNNHeads(nn.Sequential):
    _version = 2

    def __init__(self, in_channels, layers, dilation, norm_layer: Optional[Callable[..., nn.Module]] = None):
        """
        Args:
            in_channels (int): number of input channels
            layers (list): feature dimensions of each FCN layer
            dilation (int): dilation rate of kernel
            norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
        """
        blocks = []
        next_feature = in_channels
        for layer_features in layers:
            blocks.append(
                misc_nn_ops.Conv2dNormActivation(
                    next_feature,
                    layer_features,
                    kernel_size=3,
                    stride=1,
                    padding=dilation,
                    dilation=dilation,
                    norm_layer=norm_layer,
                )
            )
            next_feature = layer_features

        super().__init__(*blocks)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            num_blocks = len(self)
            for i in range(num_blocks):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}mask_fcn{i+1}.{type}"
                    new_key = f"{prefix}{i}.0.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class MaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, dim_reduced, num_classes):
        super().__init__(
            OrderedDict(
                [
                    ("conv5_mask", nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("mask_fcn_logits", nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)),
                ]
            )
        )

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            # elif "bias" in name:
            #     nn.init.constant_(param, 0)

_COMMON_META = {
    "categories": _COCO_CATEGORIES,
    "min_size": (1, 1),
}

class MaskRCNN_ResNet50_FPN_V2_Weights(WeightsEnum):
    COCO_V1 = Weights(
        url="https://download.pytorch.org/models/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth",
        transforms=ObjectDetection,
        meta={
            **_COMMON_META,
            "num_params": 46359409,
            "recipe": "https://github.com/pytorch/vision/pull/5773",
            "_metrics": {
                "COCO-val2017": {
                    "box_map": 47.4,
                    "mask_map": 41.8,
                }
            },
            "_ops": 333.577,
            "_file_size": 177.219,
            "_docs": """These weights were produced using an enhanced training recipe to boost the model accuracy.""",
        },
    )
    DEFAULT = COCO_V1


@handle_legacy_interface(
    weights=("pretrained", MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1),
    weights_backbone=("pretrained_backbone", ResNet50_Weights.IMAGENET1K_V1),
)
def maskrcnn_resnet50_fpn(
    *,
    weights: None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    weights_backbone: Optional[ResNet50_Weights] = None,
    trainable_backbone_layers: Optional[int] = None,
    # custom iou
    rpn_fg_iou_thresh=0.5,
    rpn_bg_iou_thresh=0.5,
    c_rpn_nms = 0.5,
    c_box_nms = 0.2,
    c_anchor = True,
    **kwargs: Any,
) -> MaskRCNN:
    """custom wrapper for PyTorch maskrcnn_resnet50_fpn_v2
        change ROI_generator and hyperparameters.
    """
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.verify(weights)
    weights_backbone = ResNet50_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param("num_classes", num_classes, len(weights.meta["categories"]))
    elif num_classes is None:
        num_classes = 91

    is_trained = weights is not None or weights_backbone is not None
    trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)

    backbone = resnet50(weights=weights_backbone, progress=progress)
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers, norm_layer=nn.BatchNorm2d)
    rpn_anchor_generator = _default_anchorgen()
    rpn_head = RPNHead(backbone.out_channels, rpn_anchor_generator.num_anchors_per_location()[0], conv_depth=2)
    box_head = FastRCNNConvFCHead(
        (backbone.out_channels, 7, 7), [256, 256, 256, 256], [1024], norm_layer=nn.BatchNorm2d
    )
    mask_head = MaskRCNNHeads(backbone.out_channels, [256, 256, 256, 256], 1, norm_layer=nn.BatchNorm2d)
    model = MaskRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=rpn_anchor_generator,
        rpn_head=rpn_head,
        box_head=box_head,
        mask_head=mask_head,
        **kwargs,
    )

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    ##############################################################
    # after load pre-trained weight, custom model parts load here#
    ##############################################################

    #create a custom anchor_generator for the FPN
    if c_anchor:
        sizes = ((4,), (8,), (16,), (32,), (64,))
        aspect_ratios = ((0.25, 0.5, 1.0, 2.0, 4.0),) * len(sizes)
        anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratios)

        model.rpn.anchor_generator = anchor_generator

        # 256 because that's the number of features that FPN returns
        model.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])
    
    # custom RPN NMS
    model.rpn.rpn_nms_thresh = c_rpn_nms

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # number of class in our dataset
    num_classes = 2

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # custom box NMS
    model.roi_heads.box_nms_thresh = c_box_nms

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                    hidden_layer,num_classes)
    return model


def maskrcnn_mobile(
    *,
    num_classes: Optional[int] = None,
    **kwargs: Any,
) -> MaskRCNN:
    """custom wrapper for PyTorch maskrcnn_resnet50_fpn_v2
        change ROI_generator and hyperparameters.
    """
    backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
    # ``FasterRCNN`` needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    # so we need to add it here
    backbone.out_channels = 1280

    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # ``OrderedDict[Tensor]``, and in ``featmap_names`` you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )
    
    mask_head = MaskRCNNHeads(backbone.out_channels, [256, 256, 256, 256], 1, norm_layer=nn.BatchNorm2d)

    # put the pieces together inside a Mask-RCNN model
    model = MaskRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        mask_head=mask_head,
        **kwargs,
    )
    return model

def maskrcnn_resnet(name, num_classes, pretrained, res='normal'):
    print('Using maskrcnn with {} backbone...'.format(name))

    backbone = resnet_fpn_backbone(name, pretrained=pretrained, trainable_layers=5)


    sizes = ((4,), (8,), (16,), (32,), (64,))
    aspect_ratios = ((0.25, 0.5, 1.0, 2.0, 4.0),) * len(sizes)
    anchor_generator = AnchorGenerator(
        sizes=sizes, aspect_ratios=aspect_ratios
    )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
                                                    output_size=7, sampling_ratio=2)


    model = MaskRCNN(backbone, num_classes=num_classes,
                    rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)

    model.roi_heads.box_nms_thresh = 0.2
    model.rpn.rpn_nms_thresh = 0.5
    return model


import torch, gc
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from timm import create_model

import torch.nn as nn

class SwinBackboneWrapper(nn.ModuleDict):
    def __init__(self, model):
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module

        super().__init__(layers)


from timm.models.swin_transformer import SwinTransformer
import torch.nn as nn

class SwinBackbonePermute(SwinTransformer):
    def __init__(self, *args, **kwargs):
        super(SwinBackbonePermute, self).__init__(*args, **kwargs)

    def forward_features(self, x):
        features = super().forward_features(x)  # Get feature maps from the original method

        # Permute each feature map to [B, C, W, H]
        permuted_features = {
            key: feature.permute(0, 3, 1, 2) for key, feature in features.items()
        }
        return permuted_features

import warnings
from typing import Callable, Dict, List, Optional, Union
from collections import OrderedDict

from torch import nn, Tensor
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock, FeaturePyramidNetwork, LastLevelMaxPool

class SwinIntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """

    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x.permute(0, 3, 1, 2)
        return out

class SwinBackboneWithFPN(nn.Module):
    """
    Adds a FPN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediateLayerGetter apply here.
    Args:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.
        norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
    Attributes:
        out_channels (int): the number of channels in the FPN
    """

    def __init__(
        self,
        backbone: nn.Module,
        return_layers: Dict[str, str],
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks: Optional[ExtraFPNBlock] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.body = SwinIntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
            norm_layer=norm_layer,
        )
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.body(x)
        x = self.fpn(x)
        return x


def maskrcnn_swin(name='swin', num_classes=2):
    print('Using maskrcnn with {} backbone...'.format(name))

    # Load pre-trained Swin Transformer from timm
    swin_backbone = create_model('swin_base_patch4_window7_224', pretrained=True, features_only=True)
    # swin_backbone = torchvision.models.swin_t().features

    # Check output channels from Swin Transformer to configure the FPN
    print(swin_backbone.feature_info.channels())  # Example: [128, 256, 512, 1024]

    # Create FPN from Swin Transformer
    in_channels_list = swin_backbone.feature_info.channels()  # Input channels for FPN
    out_channels = 256  # Desired output channels for FPN layers

    returned_layers = [0, 1, 2, 3]
    return_layers = {f"layers_{k}": str(v) for v, k in enumerate(returned_layers)}

    # Define the backbone with FPN
    backbone = SwinBackboneWithFPN(
        swin_backbone,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=out_channels,
    )

    # custom anchor
    sizes = ((4,), (8,), (16,), (32,), (64,))
    aspect_ratios = ((0.25, 0.5, 1.0, 2.0, 4.0),) * len(sizes)
    anchor_generator = AnchorGenerator(
        sizes=sizes, aspect_ratios=aspect_ratios
    )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
                                                    output_size=7, sampling_ratio=2)


    model = MaskRCNN(backbone, min_size=224, num_classes=num_classes,
                    rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler,
                    # custom iou
                    rpn_fg_iou_thresh=0.5,
                    rpn_bg_iou_thresh=0.5)

    model.roi_heads.box_nms_thresh = 0.2
    model.rpn.rpn_nms_thresh = 0.5
    return model


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model