<<<<<<< HEAD
import torch, pdb
import torchvision
import torch.nn.modules
from IPython import embed
from model.hubconf import *
# from hubconf import *

class mobilenet(torch.nn.Module):
    def __init__(self, backbone, pretrained = False):
        super(mobilenet, self).__init__()
        features = list(mobilenet_v2(pretrained=pretrained).features.children())
        self.features = torch.nn.Sequential(*features)
    
    def forward(self, x):
        return self.features(x)

class squeezenet(torch.nn.Module):
    def __init__(self, backbone, pretrained = False):
        super(squeezenet, self).__init__()
        features = list(squeezenet1_0(pretrained=pretrained).features.children())
        self.features = torch.nn.Sequential(*features)
    
    def forward(self, x):
        return self.features(x)

class vgg16bn(torch.nn.Module):
    def __init__(self,pretrained = False):
        super(vgg16bn,self).__init__()
        model = list(torchvision.models.vgg16_bn(pretrained=pretrained).features.children())
        model = model[:33]+model[34:43]
        self.model = torch.nn.Sequential(*model)
        
    def forward(self,x):
        return self.model(x)
        
class resnet(torch.nn.Module):
    def __init__(self, layers, pretrained = False):
        super(resnet,self).__init__()
        if layers == '18':
            model = torchvision.models.resnet18(pretrained=pretrained)
        elif layers == '34':
            model = torchvision.models.resnet34(pretrained=pretrained)
        elif layers == '50':
            model = torchvision.models.resnet50(pretrained=pretrained)
        else:
            raise NotImplementedError
        
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x4
        
class VisionTransformer(torch.nn.Module):
    def __init__(self, layers, pretrained = False):
        super(VisionTransformer, self).__init__()
        model = vit_b_16(pretrained=pretrained)
        
        self.class_token = model.class_token
        self.encoder = model.encoder
        self.image_size = model.image_size
        self.conv_proj = model.conv_proj
        self.hidden_dim = model.hidden_dim
        self.patch_size = model.patch_size

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.adaptive_avg_pool2d(x, (224, 224))

        n, c, h, w = x.shape
        p = self.patch_size

        torch._assert(h == self.image_size, "Wrong image height!")
        torch._assert(w == self.image_size, "Wrong image width!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        # x = self.heads(x)

        return x

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    model = VisionTransformer('vit_b_16').cuda()
    x = torch.zeros((1,3,288,800)).cuda() + 1
    out = model(x)
=======
import torch, pdb
import torchvision
import torch.nn.modules
from IPython import embed
from model.hubconf import *
# from hubconf import *

class mobilenet(torch.nn.Module):
    def __init__(self, backbone, pretrained = False):
        super(mobilenet, self).__init__()
        features = list(mobilenet_v2(pretrained=pretrained).features.children())
        self.features = torch.nn.Sequential(*features)
    
    def forward(self, x):
        return self.features(x)

class squeezenet(torch.nn.Module):
    def __init__(self, backbone, pretrained = False):
        super(squeezenet, self).__init__()
        features = list(squeezenet1_0(pretrained=pretrained).features.children())
        self.features = torch.nn.Sequential(*features)
    
    def forward(self, x):
        return self.features(x)

class vgg16bn(torch.nn.Module):
    def __init__(self,pretrained = False):
        super(vgg16bn,self).__init__()
        model = list(torchvision.models.vgg16_bn(pretrained=pretrained).features.children())
        model = model[:33]+model[34:43]
        self.model = torch.nn.Sequential(*model)
        
    def forward(self,x):
        return self.model(x)
        
class resnet(torch.nn.Module):
    def __init__(self, layers, pretrained = False):
        super(resnet,self).__init__()
        if layers == '18':
            model = torchvision.models.resnet18(pretrained=pretrained)
        elif layers == '34':
            model = torchvision.models.resnet34(pretrained=pretrained)
        elif layers == '50':
            model = torchvision.models.resnet50(pretrained=pretrained)
        else:
            raise NotImplementedError
        
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x4
        
class VisionTransformer(torch.nn.Module):
    def __init__(self, layers, pretrained = False):
        super(VisionTransformer, self).__init__()
        model = vit_b_16(pretrained=pretrained)
        
        self.class_token = model.class_token
        self.encoder = model.encoder
        self.image_size = model.image_size
        self.conv_proj = model.conv_proj
        self.hidden_dim = model.hidden_dim
        self.patch_size = model.patch_size

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.adaptive_avg_pool2d(x, (224, 224))

        n, c, h, w = x.shape
        p = self.patch_size

        torch._assert(h == self.image_size, "Wrong image height!")
        torch._assert(w == self.image_size, "Wrong image width!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        # x = self.heads(x)

        return x

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    model = VisionTransformer('vit_b_16').cuda()
    x = torch.zeros((1,3,288,800)).cuda() + 1
    out = model(x)
>>>>>>> d175ba8a15a74cff363e8da114147f44311bfb42
    print(out.size())