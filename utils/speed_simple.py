import torch
import time, sys
import numpy as np
sys.path.append("..")
from model.model import parsingNet
# from segmentation.model_seg import parsingNet

torch.backends.cudnn.benchmark = True
net = parsingNet(pretrained = False, backbone='34', cls_dim=(200, 52, 4)).cuda()
net.eval()

x = torch.zeros((1,3,288,800)).cuda() + 1
for i in range(10):
    y = net(x)

t_all = []
for i in range(300):
    t1 = time.time()
    y = net(x)
    t2 = time.time()
    t_all.append(t2 - t1)

print('average time:', np.mean(t_all) / 1)
print('average fps:',1 / np.mean(t_all))

print('fastest time:', min(t_all) / 1)
print('fastest fps:',1 / min(t_all))

print('slowest time:', max(t_all) / 1)
print('slowest fps:',1 / max(t_all))

from torchinfo import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
summary(net, input_size=(1, 3, 288, 800))

from torchstat import stat
stat(net.cpu(), (3, 288, 800))