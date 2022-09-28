import torch, os, cv2, sys
import scipy.special, tqdm
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

sys.path.append("..")

from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
from utils.evaluation import grid_2_inter
from IPython import embed

color_list = [(0,0,225), (255,0,0), (0,225,0), (255,0,225), (255,255,225), (0,255,255), (255,255,0), (125,255,255)]
thickness_list = [1, 3, 5, 7, 9, 11, 13, 15]
thickness_list.reverse()

raildb_row_anchor = [200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320,
                     330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450,
                     460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580,
                     590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710]
griding_num = 200

def deploy_image(file_name, net):
    frame = cv2.imread(file_name)
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    image = img_transforms(image).unsqueeze(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = image.to(device)
    net.to(device)
    outs = net(image)
    
    vis = cv2.resize(frame, (1280, 720))
    preds_inter = [grid_2_inter(out, griding_num) for out in outs]
        
    for i in range(preds_inter[0].shape[0]):
        points = [[int(x),int(y)] for (x,y) in zip(preds_inter[0][i], raildb_row_anchor) if x>=0]
        cv2.polylines(vis, (np.asarray([points])).astype(np.int32), False, color_list[i], thickness=thickness_list[i])

    cv2.imwrite(file_name[:-4]+'_output.jpg', vis)

def deploy_videos(video_path, net):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path[:-4]+'_output.avi', fourcc, 20.0, (1280, 720))

    while cap.isOpened():
        # get a frame
        ret, frame = cap.read()
        if frame is None: break

        # get a prediction
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_transforms = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])
        image = img_transforms(image).unsqueeze(0)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        image = image.to(device)
        net.to(device)
        output = net(image)[0]

        # plot a prediction
        vis = cv2.resize(frame, (1280, 720))
        preds_inter = grid_2_inter(output, griding_num)
        for i in range(preds_inter.shape[0]):
            points = [[int(x),int(y)] for (x,y) in zip(preds_inter[i], raildb_row_anchor) if x>=0]
            cv2.polylines(vis, (np.asarray([points])).astype(np.int32), False, color_list[i], thickness=thickness_list[i])

        # show and save a frame 
        # cv2.imshow("capture", vis)
        out.write(vis)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    
    out.release()        
    cap.release()
    cv2.destroyAllWindows() 
    

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    dist_print('start testing...')

    net = parsingNet(pretrained = False, backbone='18', cls_dim = (griding_num+1, len(raildb_row_anchor), 4),).cuda()

    state_dict = torch.load('../best_model.pth', map_location='cpu')
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()
    deploy_videos('./example.mp4', net)
    # deploy_image('./example.jpg', net)

# python deploy.py
