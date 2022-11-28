import torch.nn.functional as F
import sys
from util.dataset import trainloader
from util import viz
import config as c
import modules.Unet_common as common
import warnings
from torchvision import models
import torchvision.transforms as transforms
from util.vgg_loss import VGGLoss
import time
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image
from args import get_args_parser
from util.utils import *
from model.model import *

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#####################
# Model initialize: #
#####################
args = get_args_parser()
INN_net = Model().to(device)
init_model(INN_net)
INN_net = torch.nn.DataParallel(INN_net,device_ids=[0])
para = get_parameter_number(INN_net)
print(para)
params_trainable = (list(filter(lambda p: p.requires_grad, INN_net.parameters())))
optim1 = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim1, c.weight_step, gamma=c.gamma)
optim_init =optim1.state_dict()
dwt = common.DWT()
iwt = common.IWT()

class_idx = json.load(open("./util/imagenet_class_index.json"))
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
class2label = [class_idx[str(k)][0] for k in range(len(class_idx))]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()])

transform_toTensor = transforms.Compose([transforms.ToTensor()])

norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

if args.models == 'Resnet50':
    model = nn.Sequential(
        norm_layer,
        models.resnet50(pretrained=True)
    ).to(device)
elif args.models == 'Inception_v3':
    model = nn.Sequential(
        norm_layer,
        models.inception_v3(pretrained=True)
    ).to(device)
elif args.models == 'Densenet121':
    model = nn.Sequential(
        norm_layer,
        models.densenet121(pretrained=True)
    ).to(device)
else:
    sys.exit("Please choose Resnet50 or Inception_v3 or Densenet121")
model = model.eval()

try:
    totalTime = time.time()
    vgg_loss = VGGLoss(3, 1, False)
    vgg_loss.to(device)
    failnum = 0
    count = 0.0
    for i_batch, mydata in enumerate(trainloader):
        start_time = time.time()
        X_1 = torch.full((1, 3, 224, 224), 0.5).to(device)
        X_ori = X_1.to(device)
        X_ori = Variable(X_ori, requires_grad=True)
        optim2 = torch.optim.Adam([X_ori], lr=c.lr2)
        if c.pretrain:
            load(args.pre_model, INN_net)
            optim1.load_state_dict(optim_init)
        data = mydata[0].to(device)
        data = data.squeeze(0)
        lablist1 = mydata[1]
        lablist2 = mydata[2]
        n1 = int(lablist1)
        n2 = int(lablist2)
        i1 = np.array([n1])
        i2 = np.array([n2])
        source_name = index(n1)
        target_name = index(n2)
        labels = torch.from_numpy(i2).to(device)
        labels = labels.to(torch.int64).to(device)
        cover = data.to(device)  # channels = 3
        cover_dwt_1 = dwt(cover).to(device)  # channels = 12
        cover_dwt_low = cover_dwt_1.narrow(1, 0, c.channels_in).to(device)  # channels = 3

        if not os.path.exists(args.outputpath + source_name + "-" + target_name):
            os.makedirs(args.outputpath + source_name + "-" + target_name)
        save_image(cover, args.outputpath + source_name + "-" + target_name + '\\cover.png')
        for i_epoch in range(c.epochs):
            #################
            #    train:   #
            #################
            CGT = X_ori.to(device)
            CGT_dwt_1 = dwt(CGT).to(device)# channels =12
            CGT_dwt_low_1 = CGT_dwt_1.narrow(1, 0, c.channels_in).to(device)# channels =3
            input_dwt_1 = torch.cat((cover_dwt_1, CGT_dwt_1), 1).to(device)  # channels = 12*2
            output_dwt_1 = INN_net(input_dwt_1).to(device)  # channels = 24
            output_steg_dwt_2 = output_dwt_1.narrow(1, 0, 4 * c.channels_in).to(device)  # channels = 12
            output_step_low_2 = output_steg_dwt_2.narrow(1, 0, c.channels_in ).to(device)  # channels = 3

            output_steg_dwt_low_1 = output_steg_dwt_2.narrow(1, 0, c.channels_in).to(device)  # channels = 3

            output_r_dwt_1 = output_dwt_1.narrow(1, 4 * c.channels_in, 4 * c.channels_in).to(device)
            # get steg
            output_steg_1 = iwt(output_steg_dwt_2).to(device)  # channels = 3

            output_r = iwt(output_r_dwt_1).to(device)

            output_steg_1 = torch.clamp(output_steg_1, min=0, max=1).to(device)

            eta = torch.clamp(output_steg_1 - cover, min=-args.eps, max=args.eps)
            output_steg_1 = torch.clamp(cover + eta, min=0, max=1)

            #################
            #     loss:     #
            #################
            g_loss = guide_loss(output_steg_1.cuda(), cover.cuda()).to(device)

            vgg_on_cov = vgg_loss(cover).to(device)
            vgg_on_steg_1 = vgg_loss(output_steg_1).to(device)

            perc_loss = guide_loss(vgg_on_cov, vgg_on_steg_1).to(device)

            l_loss = guide_loss(output_step_low_2.cuda(), cover_dwt_low.cuda()).to(device)

            out = model(output_steg_1 * 255.0).to(device)
            _, pre = torch.max(out.data, 1)
            percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
            _, indices = torch.sort(out, descending=True)
            adv_cost = nn.CrossEntropyLoss().to(device)
            adv_loss = adv_cost(out, labels).to(device)

            suc_rate = ((pre == labels).sum()).cpu().detach().numpy()

            total_loss = c.lamda_guide * g_loss + c.lamda_low_frequency * l_loss + args.lamda_adv * adv_loss + c.lamda_per * perc_loss
            ii = int(pre)
            state = "img" + str(i_batch) + ":" + str(suc_rate)
            #################
            #     Exit:     #
            #################
            if suc_rate == 1:
                if (int(percentage[indices[0]][0]) >= 85):
                    print(state)
                    print([(class2label[idx], percentage[idx].item()) for idx in indices[0][:5]])
                    save_image(output_steg_1, args.outputpath + source_name + "-" + target_name +'\\'+ str(i_epoch) + 'result.png')
                    output_r = normal_r(output_r)
                    save_image(output_r, args.outputpath + source_name + "-" + target_name + '\\r.png')
                    count +=1
                    break
                if (i_epoch >= 2000):
                    print([(class2label[idx], percentage[idx].item()) for idx in indices[0][:5]])
                    save_image(output_steg_1, args.outputpath + source_name + "-" + target_name + '\\'+
                               str(i_epoch) + "_" + str(int(percentage[indices[0]][0])) +'d_result.png')
                    output_r = normal_r(output_r)
                    save_image(output_r , args.outputpath + source_name + "-" + target_name + '\\r.png')
                    count +=1
                    break
            if (i_epoch >= 5000):
                failnum += 1
                print([(class2label[idx], percentage[idx].item()) for idx in indices[0][:5]])
                save_image(output_steg_1 , args.outputpath + source_name + "-" + target_name + '\\' +
                           str(i_epoch) + 'dw_result.png')
                output_r = normal_r(output_r)
                save_image(output_r , args.outputpath + source_name + "-" + target_name + '\\r.png')
                count +=1
                break

            #################
            #   Backward:   #
            #################
            optim1.zero_grad()
            optim2.zero_grad()
            total_loss.backward()
            optim1.step()

            C_out = model(CGT * 255.0).to(device)
            C_adv_loss = adv_cost(C_out, labels).to(device)
            C_adv_loss.backward()
            optim2.step()

            weight_scheduler.step()
            lr_min = c.lr_min
            lr_now = optim1.param_groups[0]['lr']
            if lr_now < lr_min:
                optim1.param_groups[0]['lr'] = lr_min
        save_image(CGT , args.outputpath + source_name + "-" + target_name + '\\CGT.png')
    totalstop_time = time.time()
    time_cost = totalstop_time - totalTime
    Total_suc_rate = (count-failnum)/count
    print("Total cost time :" + str(time_cost))
    print("Total suc rate :" + str(Total_suc_rate))
except:
    raise

finally:
    viz.signal_stop()
