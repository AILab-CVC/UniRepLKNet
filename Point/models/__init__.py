from models.layers import HorNet
from timm.models import convnext, swin_transformer
from timm.models.registry import register_model
from torchvision import models

# init convnext that not defined in the original timm
convnext.default_cfgs['convnext_tiny_in22k']  = convnext._cfg(
        url="https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth", num_classes=21841)
convnext.default_cfgs['convnext_small_in22k']  = convnext._cfg(
        url="https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth", num_classes=21841)

@register_model
def convnext_tiny_in22k(pretrained=False, **kwargs):
    model_args = dict(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), **kwargs)
    model = convnext._create_convnext('convnext_tiny_in22k', pretrained=pretrained, **model_args)
    return model

@register_model
def convnext_small_in22k(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    model = convnext._create_convnext('convnext_small_in22k', pretrained=pretrained, **model_args)
    return model

# init swin that not defined in the original timm
# in22k
swin_transformer.default_cfgs['swin_tiny_patch4_window7_224_in22k']  = swin_transformer._cfg(
    url='https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22k.pth',
    num_classes=21841) # 80.9w
swin_transformer.default_cfgs['swin_small_patch4_window7_224_in22k']  = swin_transformer._cfg(
    url='https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_small_patch4_window7_224_22k.pth',
    num_classes=21841) # 83.2
# in1k
swin_transformer.default_cfgs['swin_tiny_patch4_window7_224_in1k']  = swin_transformer._cfg(
    url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth')  # 81.2
swin_transformer.default_cfgs['swin_small_patch4_window7_224_in1k']  = swin_transformer._cfg( 
    url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth') # 83.2
swin_transformer.default_cfgs['swin_base_patch4_window7_224_in1k']  = swin_transformer._cfg(
    url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth')  # 83.5

@register_model
def swin_tiny_patch4_window7_224_in22k(pretrained=False, **kwargs):
    """ Swin-T @ 224x224, trained ImageNet-22k
    """
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), **kwargs)
    return swin_transformer._create_swin_transformer('swin_tiny_patch4_window7_224_in22k', pretrained=pretrained, **model_kwargs)

@register_model
def swin_small_patch4_window7_224_in22k(pretrained=False, **kwargs):
    """ Swin-S @ 224x224, trained ImageNet-22k
    """
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24), **kwargs)
    return swin_transformer._create_swin_transformer('swin_small_patch4_window7_224_in22k', pretrained=pretrained, **model_kwargs)

@register_model
def swin_tiny_patch4_window7_224_in1k(pretrained=False, **kwargs):
    """ Swin-T @ 224x224, trained ImageNet-1k
    """
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), **kwargs)
    return swin_transformer._create_swin_transformer('swin_tiny_patch4_window7_224_in1k', pretrained=pretrained, **model_kwargs)

@register_model
def swin_small_patch4_window7_224_in1k(pretrained=False, **kwargs):
    """ Swin-S @ 224x224, trained ImageNet-1k
    """
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24), **kwargs)
    return swin_transformer._create_swin_transformer('swin_small_patch4_window7_224_in1k', pretrained=pretrained, **model_kwargs)

@register_model
def swin_base_patch4_window7_224_in1k(pretrained=False, **kwargs):
    """ Swin-B @ 224x224, trained ImageNet-1k
    """
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), **kwargs)
    return swin_transformer._create_swin_transformer('swin_base_patch4_window7_224_in1k', pretrained=pretrained, **model_kwargs)

# ResNet from torchvision
@register_model
def resnet_18_torchvision(pretrained=False, **kwargs):
    """ResNet-18 pretrained on ImageNet-1k from torchvision    
    """
    return models.resnet18(pretrained=pretrained)


@register_model
def resnet_50_torchvision(pretrained=False, **kwargs):
    """ResNet-50 pretrained on ImageNet-1k from torchvision    
    """
    return models.resnet50(pretrained=pretrained)

@register_model
def resnet_101_torchvision(pretrained=False, **kwargs):
    """ResNet-101 pretrained on ImageNet-1k from torchvision    
    """
    return models.resnet101(pretrained=pretrained)

@register_model
def resnet_152_torchvision(pretrained=False, **kwargs):
    """ResNet-152 pretrained on ImageNet-1k from torchvision    
    """
    return models.resnet152(pretrained=pretrained)
