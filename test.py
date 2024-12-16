import argparse
import math
import lpips
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from dataset import kodak
import Model.JND_LIC as model
import Util.torch_msssim as torch_msssim
from Model.jnd_context_atten import Weighted_Gaussian




class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class JNDloss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=0.01, metric="mse", return_type="all"):
        super().__init__()
        self.lmbda = lmbda
        self.return_type = return_type

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        return out


class rdloss(nn.Module):
    def __init__(self, lmbda):
        super(rdloss, self).__init__()
        self.mse_func = nn.MSELoss()
        self.loss_fun_lpips = lpips.LPIPS(net='alex')
        self.jnd_loss_fun = JNDloss()
        self.lmbda = lmbda

    def forward(self, epoch, input, fake, jnd_out, xp2, xp3):
        num_pixels = input.size()[0] * input.size()[2] * input.size()[3]

        mse = self.mse_func(fake*255, input*255)
        jndloss = self.jnd_loss_fun(jnd_out, fake)
        jnd_bpp = jndloss["bpp_loss"]
        lpips = self.loss_fun_lpips.forward(input, fake)
        avg_lpips = torch.mean(lpips, dim=0)
        train_bpp1 = jnd_bpp
        train_bpp2 = torch.sum(torch.log(xp2)) / (-np.log(2) * num_pixels)
        train_bpp3 = torch.sum(torch.log(xp3)) / (-np.log(2) * num_pixels)
        all_bpp = train_bpp1 + train_bpp2 + train_bpp3
        all_loss = train_bpp1 + train_bpp2 + train_bpp3 + self.lmbda * (mse + 1000 * avg_lpips)
        return all_loss, avg_lpips, all_bpp, mse, jnd_bpp


def test_epoch (epoch,  model, context):

    model.eval().cuda()
    context.eval().cuda()
    msssim_func = torch_msssim.MS_SSIM(max_val=1.).cuda()
    all_loss_func = rdloss(args.lmbda).cuda()
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    JND_bpp = AverageMeter()
    PSNR = AverageMeter()
    LPIPS = AverageMeter()
    kodakdir = "E:/zgy/data/kodak/Kodak_dataset"
    jnddir = "E:/zgy/data/kodak/kodak_jnd"
    kodak_dataset = kodak(root_dir=kodakdir, jnd_dir=jnddir)
    test_dataloader = DataLoader(dataset=kodak_dataset, batch_size=1, num_workers=8, shuffle=False)

    with torch.no_grad():
        for i, d in enumerate(test_dataloader):
            batch_x = d[:, 0:3, :, :]
            jnd = d[:, 3:6, :, :]

            batch_x = Variable(batch_x).cuda()
            jnd = Variable(jnd).cuda()

            fake,  xp2, xq1, x3, jnd, jnd_out = model.forward(batch_x, jnd=jnd, if_training=1)
            xp3, _ = context.forward(xq1, x3, jnd)

            ssim = msssim_func(fake,batch_x)

            l_rec,  avg_lpips, all_bpp, mseloss1, jnd_bpp = all_loss_func(epoch, batch_x, fake, jnd_out, xp2, xp3)
            JND_bpp.update(jnd_bpp)
            PSNR.update(10*np.log10(1/mseloss1.item()))
            bpp_loss.update(all_bpp.item())
            LPIPS.update(avg_lpips.item())
            loss.update(l_rec.item())
            mse_loss.update(mseloss1.item())
    mse_loss_avg = mse_loss.avg
    loss_avg = loss.avg
    bpp_loss_avg = bpp_loss.avg
    jnd_bpp_avg = JND_bpp.avg
    PSNR_avg = PSNR.avg
    LPIPS_avg = LPIPS.avg

    print("Test epoch {}: Average losses:{} psnr:{} lpips:{}".format(epoch,loss_avg, PSNR_avg, LPIPS_avg))
    print( "bpp:{} mse_loss{} ssim_loss{} jnd_bpp:{}".format(bpp_loss_avg, mse_loss_avg, ssim, jnd_bpp_avg))

    return loss_avg, bpp_loss_avg, PSNR_avg, LPIPS_avg, mse_loss_avg


def main(args):

    txt_dir = args.out_dir + "/best.txt"
    image_comp = model.Image_coding(input_features=3, N1= 192, N2= 128, M= 192, M1=96, patch_size=2).cuda()  # ,nf,nb,gc=qp32
    context = Weighted_Gaussian(args.M).cuda()

    # 加载模型
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location="cuda:0")
        image_comp.load_state_dict(checkpoint["image_state_dict"])
        context.load_state_dict(checkpoint["context_state_dict"])
        last_epoch = int(checkpoint["epoch"])

    loss, bpp, psnr, lpips, mse = test_epoch(last_epoch, image_comp, context)
    txt_loss = str(loss)
    txt_psnr = str(psnr)
    txt_bpp = str(bpp)
    txt_step = str(last_epoch)
    txt_lpips = str(lpips)
    with open(txt_dir, 'a') as txt:
        txt.write("epoch: " + txt_step + " loss: " + txt_loss + " psnr: " + txt_psnr + " bpp: " + txt_bpp + " lpips: " + txt_lpips +'\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2024, help="the value of M")
    parser.add_argument("--M", type=int, default=192, help="the value of M")
    parser.add_argument("--N2", type=int, default=128, help="the value of N2")
    parser.add_argument("--lambda", type=float, default=1e-3 * 4, dest="lmbda", help="Lambda for rate-distortion tradeoff.")
    parser.add_argument("--lr", type=float, default=1e-4, help="initial learning rate.")
    parser.add_argument('--out_dir', type=str, default='E:/zgy/code/ours/save_model/qp16')
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument("--cuda", default=False, help="Use cuda")
    args = parser.parse_args()
    print(args)
    main(args)
