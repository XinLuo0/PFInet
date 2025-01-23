import torch
import time
import argparse
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import os
import numpy as np

from dataload import *
from model_PFInet import *

# Settings
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--angin", type=int, default=5, help="angular resolution")
    parser.add_argument('--model_name', type=str, default='LLE')
    parser.add_argument('--test_name', type=str, default='20_LLE_5x5')
    parser.add_argument('--trainset_dir', type=str, default='./Data/Train/TrainingData_20_LLE_5x5/')
    parser.add_argument('--testset_dir', type=str, default='./Data/Test/TestData_20_LLE_5x5/')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=15, help='number of epochs to update learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decaying factor')
    parser.add_argument("--patchsize", type=int, default=64, help="crop into patches for validation")
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='./log/xxx.tar')
    parser.add_argument('--save_path', type=str, default='./log/')

    return parser.parse_args() 



def train(cfg, train_loader, test_loader):
    if not os.path.exists(cfg.save_path):
        os.mkdir(cfg.save_path)

    net = Net()
    net = net.to(cfg.device)
    cudnn.benchmark = False
    epoch_state = 0


    if cfg.load_pretrain:
        if os.path.isfile(cfg.model_path):
            device = torch.device("cuda:0")  # Specify the desired CUDA device (0 in this example)
            model = torch.load(cfg.model_path, map_location=device)
            net.load_state_dict(model['state_dict'])
            epoch_state = model["epoch"]
            print("load pre-train at epoch {}".format(epoch_state))
        else:
            print("=> no model found at '{}'".format(cfg.load_model))

    # net = torch.nn.DataParallel(net, device_ids=[0, 1])
  
    criterion_ssim = SSIM(window_size = 11).to(cfg.device)
    
    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)
    scheduler._step_count = epoch_state
    loss_epoch = []
    loss_list = []

    for idx_epoch in range(epoch_state, cfg.n_epochs):
        for idx_iter, (data, label) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data, label = Variable(data).to(cfg.device), Variable(label).to(cfg.device)
            label, data = augmentation(label, data)
            b,u,v,c,h,w = label.shape
            label = label.reshape(b,u*v*c,h,w)
            # print(data.shape)
           
            phr = net(data)
            
            loss =  criterion_ssim(phr, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.data.cpu())

        if idx_epoch % 1 == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))
            print(time.ctime()[4:-5] + ' Epoch----%5d, loss---%f' % (idx_epoch + 1, float(np.array(loss_epoch).mean())))
            txtfile = open('./log/' + cfg.model_name + '_training.txt', 'a')
            txtfile.write(time.ctime()[4:-5] + ' Epoch----%5d, loss---%f' % (idx_epoch + 1, float(np.array(loss_epoch).mean())) + '\n')
            txtfile.close()
            save_ckpt({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'loss': loss_list,},
                save_path='./log/', filename=cfg.model_name + '_' + str(cfg.angin) + 'x' + str(cfg.angin) + '_epoch_' + str(idx_epoch + 1) + '.tar')
            loss_epoch = []

        ''' evaluation '''
        
        with torch.no_grad():
            psnr_testset = []
            ssim_testset = []
            
            psnr_epoch_test, ssim_epoch_test = valid(test_loader, net, cfg.angin)
            psnr_testset.append(psnr_epoch_test)
            ssim_testset.append(ssim_epoch_test)
            print(time.ctime()[4:-5] + ' Valid----%15s, PSNR---%f, SSIM---%f' % (cfg.test_name, psnr_epoch_test, ssim_epoch_test))
            txtfile = open('./log/' + cfg.model_name + '_training.txt', 'a')
            txtfile.write('Dataset----%10s,\t PSNR---%f,\t SSIM---%f\n' % (cfg.test_name, psnr_epoch_test, ssim_epoch_test))
            txtfile.close()
            pass
        pass

def valid(test_loader, net, angin):
    psnr_iter_test = []
    ssim_iter_test = []
    for idx_iter, (data, label) in (enumerate(test_loader)):
        data = data.squeeze().to(cfg.device)
        label = label.squeeze()
        sub_lfs = LFdivide(data, cfg.patchsize, cfg.patchsize//2)
        numU, numV, u, v, c, h, w = sub_lfs.shape
        minibatch = 4
        # print(numU)
        num_inference = numU*numV//minibatch
        sub_lfs =  rearrange(sub_lfs, 'n1 n2 u v c h w -> (n1 n2) u v c h w')
        
        with torch.no_grad():
            out_lf = []
            for idx_inference in range(num_inference):
                torch.cuda.empty_cache()
                tmp = sub_lfs[idx_inference*minibatch:(idx_inference+1)*minibatch,:,:,:,:,:]
                out_tmp = net(tmp.to(cfg.device))
                out_lf.append(out_tmp)
                if (numU*numV)%minibatch:
                    torch.cuda.empty_cache()
                    tmp = sub_lfs[(idx_inference+1)*minibatch:,:,:,:,:,:]
                    out_tmp = net(tmp.to(cfg.device))
                    out_lf.append(out_tmp)
        out_lfs = torch.cat(out_lf, 0)
        out_lfs = rearrange(out_lfs, '(n1 n2) (u1 u2 c) h w -> n1 n2 u1 u2 c h w', n1=numU, n2=numV, u1=angin, u2=angin)
        outLF = LFintegrate(out_lfs, cfg.patchsize, cfg.patchsize // 2)

        outLF = outLF[:,:, :, 0 : data.shape[3], 0 : data.shape[4]]
        
        psnr, ssim = cal_metrics(label, outLF)
        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)
        pass

    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())

    return psnr_epoch_test, ssim_epoch_test

def save_ckpt(state, save_path='./checkpoint', filename='checkpoint.tar'):
    torch.save(state, os.path.join(save_path,filename), _use_new_zipfile_serialization=False)

def augmentation(x, y):
    if random.random() < 0.5:  # flip along U-H direction
        x = torch.flip(x, dims=[1, 4])
        y = torch.flip(y, dims=[1, 4])
    if random.random() < 0.5:  # flip along W-V direction
        x = torch.flip(x, dims=[2, 5])
        y = torch.flip(y, dims=[2, 5])
    if random.random() < 0.5: # transpose between U-V and H-W
        x = x.permute(0, 2, 1, 3, 5, 4)
        y = y.permute(0, 2, 1, 3, 5, 4)

    return x, y

def main(cfg):
    setup_seed(10)
    train_set = DataSetLoader(dataset_dir=cfg.trainset_dir)
    train_loader = DataLoader(dataset=train_set, num_workers=8, batch_size=cfg.batch_size, shuffle=True)
    test_set = DataSetLoader(dataset_dir=cfg.testset_dir)
    test_loader = DataLoader(dataset=test_set, num_workers=8, batch_size=1, shuffle=False)
    train(cfg, train_loader, test_loader)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
  
if __name__ == '__main__':
    cfg = parse_args()    
    main(cfg)
