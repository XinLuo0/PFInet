import time
import argparse
import torch.backends.cudnn as cudnn

from tqdm import tqdm
import scipy.io as sio
import time
import imageio
import torchvision

from dataload import *
from model_PFInet import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--test_name', type=str, default='20')
    parser.add_argument("--angin", type=int, default=5, help="angular resolution")
    parser.add_argument('--testset_dir', type=str, default='./Data/Test/TestData_20_LLE_5x5/')
    parser.add_argument("--patchsize", type=int, default=64, help="LFs are cropped into patches to save GPU memory")
    parser.add_argument('--model_path', type=str, default='./log/20_LLE_5x5_PFInet.tar')
    parser.add_argument('--save_path', type=str, default='./Figs/')

    return parser.parse_args()


def test(cfg, test_loader):

    if not os.path.exists(cfg.save_path):
        os.mkdir(cfg.save_path)
        
    net = Net()
    net.to(cfg.device)
    cudnn.benchmark = True


    
    if os.path.isfile(cfg.model_path):
        
        device = torch.device("cuda:0")  # Specify the desired CUDA device (0 in this example)
        model = torch.load(cfg.model_path, map_location=device)

        net.load_state_dict(model['state_dict'])
        
    else:
        print("=> no model found at '{}'".format(cfg.load_model))

    #net = torch.nn.DataParallel(net, device_ids=[0, 1])
    with torch.no_grad():
        psnr_testset = []
        ssim_testset = []
       
        psnr_epoch_test, ssim_epoch_test = inference(test_loader, net, cfg.angin)
        psnr_testset.append(psnr_epoch_test)
        ssim_testset.append(ssim_epoch_test)
        print(time.ctime()[4:-5] + ' Valid----, PSNR---%f, SSIM---%f' % (psnr_epoch_test, ssim_epoch_test))
        
    
def inference(test_loader, net, angin):
    psnr_iter_test = []
    ssim_iter_test = []
   
    # ti = []
    for idx_iter, (data, label) in (enumerate(test_loader)):
        data = data.squeeze().to(cfg.device)
        label = label.squeeze()
        sub_lfs = LFdivide(data, cfg.patchsize, cfg.patchsize//2)
        numU, numV, u, v, c, h, w = sub_lfs.shape
        minibatch = 8
        num_inference = numU*numV//minibatch
        sub_lfs =  rearrange(sub_lfs, 'n1 n2 u v c h w -> (n1 n2) u v c h w')
        
        t1 = time.time()
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
        outLF = LFintegrate(out_lfs, cfg.patchsize, cfg.patchsize//2)

        outLF = outLF[:,:, :, 0 : data.shape[3], 0 : data.shape[4]]
        t2 = time.time()
        psnr, ssim = cal_metrics(label, outLF)
        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)
    
       
        u,v,c,h,w = outLF.shape
        outLF = outLF.contiguous().view(u*v,c,h,w)
        plot_out_pred = torchvision.utils.make_grid(outLF,nrow=5, padding=0, normalize=False)
        x = np.transpose(plot_out_pred.detach().cpu().numpy(),(1,2,0))
        plot_out_pred = (np.clip(x,0,1)*255).astype(np.uint8) 
        
        # # Save images
        imageio.imwrite('./Figs/{}_IMG_PRED.png'.format(idx_iter),plot_out_pred)

    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())

    return psnr_epoch_test, ssim_epoch_test

def main(cfg):
    test_set = DataSetLoader(dataset_dir=cfg.testset_dir)
    test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)
    test(cfg, test_loader)


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
