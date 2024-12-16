import argparse
import math
import os
import random
import lpips
import numpy
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from dataset import kodak, image
import Model.JND_LIC as model
import Util.torch_msssim as torch_msssim
from Model.jnd_context_atten import Weighted_Gaussian
import time
import gc
import tqdm



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

class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)

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


def adjust_learning_rate(optimizer, epoch, init_lr):
    if epoch < 10:
        lr = init_lr
    else:
        lr = init_lr * (0.5 ** (epoch // 30))
    if lr < 5e-6:
        lr = 5e-6
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train(args, epoch, image_comp, context, opt1, opt2):
    train_dataset = image(root_dir="D:/data/flicker_data/2w", txt_file="D:/data/flicker_data/all_name.txt")
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True, num_workers=8)

    lamb = args.lmbda
    # loss_func = torch_msssim.MS_SSIM(max_val=1.).cuda()
    msssim_func = torch_msssim.MS_SSIM(max_val=1.).cuda()
    all_loss_func = rdloss(lamb).cuda()


    last_time = time.time()

    for param_group in opt1.param_groups:
        cur_lr = param_group['lr']

    for step, d in enumerate(tqdm.tqdm(train_dataloader)):
        d = d.cuda()
        batch_x = d[:, 0:3, :, :]
        jnd = d[:, 3:6, :, :]
        batch_x = Variable(batch_x).cuda()
        jnd = Variable(jnd).cuda()
        fake, xp2, xq1, x3, jnd_for_quamtize, jnd_out = image_comp.forward(batch_x, jnd, if_training=0)
        xp3, _ = context.forward(xq1, x3, jnd_for_quamtize)


        # MS-SSIM
        # dloss = 1.0 - loss_func(fake, batch_x)
        msssim = msssim_func(fake, batch_x)

        # l_rec = lamb * avg_lpips + all_bpp
        l_rec, avg_lpips, all_bpp, mse1, jnd_bpp = all_loss_func(epoch, batch_x, fake, jnd_out, xp2, xp3)
        train_bpp1 = jnd_bpp
        opt2.zero_grad()
        l_rec.backward()
        gc.collect()
        # gradient clip
        torch.nn.utils.clip_grad_norm_(image_comp.parameters(), 5)
        torch.nn.utils.clip_grad_norm_(context.parameters(), 5)

        opt1.step()
        opt2.step()

        if step % 10 == 0:
            with open(os.path.join(args.out_dir, 'train_jnd'+str(int(args.lmbda))+'.log'), 'a') as fd:
                time_used = time.time()-last_time
                last_time = time.time()
                mse = mse1.item()
                psnr = 10.0 * np.log10(255.**2/mse1.item())
                msssim_dB = -10*np.log10(1-(msssim.item()))
                bpp_total = all_bpp.item()
                all_loss = l_rec.item()
                bpp_jnd = train_bpp1.item()
                lpips_loss = avg_lpips.item()
                fd.write('ep:%d step:%d time:%.1f lr:%.8f loss:%.6f MSE:%.6f lpips:%.4f bpp_jnd:%.4f bpp_total:%.4f psnr:%.2f msssim:%.2f\n'
                         %(epoch, step, time_used, cur_lr, all_loss, mse, lpips_loss, bpp_jnd, bpp_total, psnr, msssim_dB))
            fd.close()
            print('epoch', epoch, 'step:', step, 'LOSS:', all_loss, 'LPIPS:', lpips_loss,
                 'PSNR:', psnr, 'BPP:', bpp_total)


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

def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    torch.save(state, filename)

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    if args.seed is not None:
        random.seed(args.seed)
        numpy.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    txt_dir = args.out_dir + "/best.txt"
    image_comp = model.Image_coding(input_features=3, N1= 192, N2= 128, M= 192, M1=96, patch_size=2).cuda()  # ,nf,nb,gc=qp32
    context = Weighted_Gaussian(args.M).cuda()

    opt1 = torch.optim.Adam(image_comp.parameters(), lr=args.lr)
    opt2 = torch.optim.Adam(context.parameters(), lr=args.lr)

    last_epoch = 0
    best_loss = 100000000
    # 加载模型
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location="cuda:0")
        image_comp.load_state_dict(checkpoint["image_state_dict"])
        context.load_state_dict(checkpoint["context_state_dict"])
        last_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        opt1.load_state_dict(checkpoint["optimizer"])
        opt2.load_state_dict(checkpoint["aux_optimizer"])


    for epoch in range(last_epoch, 200):

        a = opt1.param_groups[0]['lr']
        print(f"Learning rate: {a}")
        train(args, epoch, image_comp, context, opt1, opt2)
        adjust_learning_rate(optimizer=opt1, epoch=epoch, init_lr=args.lr)
        loss, bpp, psnr, lpips, mse = test_epoch(epoch, image_comp, context)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        save_checkpoint(
            {
                "epoch": epoch,
                "context_state_dict": context.state_dict(),
                "image_state_dict": image_comp.state_dict(),
                "best_loss": best_loss,
                "optimizer": opt1.state_dict(),
                "aux_optimizer": opt2.state_dict(),
            },
            is_best,
            filename=f"{args.out_dir}/{epoch}_checkpoint.pth",
        )
        if is_best:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "context_state_dict": context.state_dict(),
                    "image_state_dict": image_comp.state_dict(),
                    "best_loss": best_loss,
                    "optimizer": opt1.state_dict(),
                    "aux_optimizer": opt2.state_dict(),
                },
                is_best,
                filename=f"{args.out_dir}/best_checkpoint.pth",
            )
            txt_loss = str(loss)
            txt_psnr = str(psnr)
            txt_bpp = str(bpp)
            txt_step = str(epoch)
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
