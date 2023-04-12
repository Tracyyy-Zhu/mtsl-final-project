import torch
from torch import nn

class ResNet(nn.Module):
    # Basic version for now
    def __init__(self, model_name, num_class):
        super(ResNet,self).__init__()
        
        self.model = init_pretrained_model(model_name)
        self.num_class = num_class
        
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, self.num_class)
    
    def forward(self, x):
        return self.model(x)
    
def init_pretrained_model(model_name: str):
    """
    Loading pretrained model base on model name
    """
    
    if model_name == "resnet50":
        return torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")
    else:
        raise Exception(f"Model {model_name} not supported yet.")
