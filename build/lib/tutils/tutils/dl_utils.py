"""
    Utils for DL models
"""

def count_params(model):
    """
        Pytorch Model
    """
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    return pytorch_total_params


def count_model(model, example_input):
    """
        Use API from THOP: PyTorch-OpCounter (https://github.com/Lyken17/pytorch-OpCounter)
            pip install thop
        Or:
            pip install --upgrade git+https://github.com/Lyken17/pytorch-OpCounter.git
        
        Basic Usage:
        from torchvision.models import resnet50
        from thop import profile
        model = resnet50()
        input = torch.randn(1, 3, 224, 224)
        macs, params = profile(model, inputs=(input, ))

        for more details, refer to THOP!
    """
    from thop import profile
    macs, params = profile(model, inputs=(example_input, ))
    return {"macs": macs, "params": params}