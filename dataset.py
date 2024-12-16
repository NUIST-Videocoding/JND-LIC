import random
import re
import cv2
import numpy
from torch.utils.data import Dataset
import torch
import os

class kodak(Dataset):
    def __init__(self,root_dir,jnd_dir):
        super(kodak, self).__init__()
        self.root_dir = root_dir
        self.jnd_dir = jnd_dir
        self.list = sorted(os.listdir(root_dir))
        self.list.sort(key=lambda x: int(x.replace("kodim", "").split('.')[0]))
        self.jnd_list = os.listdir(jnd_dir)
        self.jnd_list.sort(key=lambda x: int(x.replace("kodim", "").split('.')[0]))

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        root = os.path.join(self.root_dir + "/" + self.list[idx])
        jnd_root = os.path.join(self.jnd_dir + "/" + self.jnd_list[idx])
        img = cv2.imread(root)
        jnd = numpy.load(jnd_root)
        (h, w, c) = img.shape
        # crop = img[0:256, 0:256, :] / 255
        crop = img/255
        crop = torch.Tensor(crop)
        # jnd_crop = jnd[0:256, 0:256, :]
        jnd_crop = jnd
        jnd_crop = torch.Tensor(jnd_crop)
        sample = torch.cat([crop, jnd_crop], dim=2)
        sample = sample.permute(2, 0, 1)
        return sample

class kodak1(Dataset):
    def __init__(self,root_dir,jnd_dir):
        super(kodak1, self).__init__()
        self.root_dir = root_dir
        self.jnd_dir = jnd_dir
        self.list = sorted(os.listdir(root_dir))
        self.list.sort(key=lambda x: int(x.replace("kodim", "").split('.')[0]))
        self.jnd_list = os.listdir(jnd_dir)
        self.jnd_list.sort(key=lambda x: int(x.replace("kodim", "").split('.')[0]))

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        root = os.path.join(self.root_dir + "/" + self.list[idx])
        jnd_root = os.path.join(self.jnd_dir + "/" + self.jnd_list[idx])
        split_strings = re.split(r'\.', self.list[idx])
        name = split_strings[0]
        img = cv2.imread(root)
        jnd = cv2.imread(jnd_root)
        (h, w, c) = img.shape
        crop = img/255
        crop = torch.Tensor(crop)
        jnd_crop = jnd
        jnd_crop = torch.Tensor(jnd_crop)
        sample = torch.cat([crop, jnd_crop], dim=2)
        sample = sample.permute(2, 0, 1)
        return sample, name

class kodak_image(Dataset):
    def __init__(self,root_dir,jnd_dir):
        super(kodak_image, self).__init__()
        self.root_dir = root_dir

        self.list = os.listdir(root_dir)
        self.list.sort(key=lambda x: int(x.replace("kodim", "").split('.')[0]))


    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        root = os.path.join(self.root_dir + "/" + self.list[idx])
        img = cv2.imread(root)
        # crop = img[0:256, 0:256, :] / 255
        crop = img/255
        crop = torch.Tensor(crop)
        # jnd_crop = jnd[0:256, 0:256, :]
        sample = crop.permute(2, 0, 1)
        return sample

class image(Dataset):
    def __init__(self, txt_file, root_dir):
        super(image, self).__init__()
        f = open(txt_file, "r")
        self.list = f.readlines()
        a = len(self.list)
        for i in range(a):
            self.list[i] = self.list[i].strip().replace('\n', '')#去掉列表中每一个元素的换行符
        f.close()
        self.root_dir = root_dir

    def __len__(self):
        return len(self.list)

    def read_img(self,rootdir):
        img = cv2.imread(rootdir + "/1.jpg")
        jnd = numpy.load(rootdir + "/1.npz")
        jnd = jnd['arr_0']
        (h, w, c) = img.shape
        h_ran = numpy.random.randint(0, (h - 256))
        w_ran = numpy.random.randint(0, (w - 256))
        crop = img[h_ran:h_ran + 256, w_ran:w_ran + 256, :]
        jnd_crop = jnd[h_ran:h_ran + 256, w_ran:w_ran + 256, :]
        crop = torch.Tensor(crop)
        jnd_crop = torch.Tensor(jnd_crop)
        crop = crop/255
        sample = torch.cat([crop, jnd_crop], dim=2)
        sample = sample.permute(2, 0, 1)
        return sample

    def __getitem__(self,idx): # 负责按索引取出某个数据，并对该数据做预处理
        root = self.root_dir+ "/"+self.list[idx]
        img = self.read_img(rootdir=root)
        return img

class image_1(Dataset):
    def __init__(self, root_dir, jnd_dir):
        # super(image_, self).__init__()
        # self.root_dir = root_dir
        self.dir_lists = os.listdir(root_dir)
        self.filename_lists = []
        self.jnd_filename_lists = []
        for dir in self.dir_lists:
            filenames = os.listdir(os.path.join(root_dir,dir))
            select = random.sample(filenames,k=300)
            for file in select:
                split_strings = re.split(r'\.', file)
                jnd_file = split_strings[0] + ".png"
                self.filename_lists.append(os.path.join(os.path.join(root_dir,dir), file))
                self.jnd_filename_lists.append(os.path.join(os.path.join(jnd_dir,dir), jnd_file))

        # print(self.filename_lists)

        # self.transform = transforms

    def __len__(self):
        return len(self.filename_lists)

    def read_img(self, rootdir, jnd_dir):
        # img = Image.open(rootdir).convert("RGB")
        img = cv2.imread(rootdir)
        jnd = cv2.imread(jnd_dir)
        
        # crop = self.transform(img)
        # sample = crop.permute(2, 0, 1)
        return img, jnd

    def __getitem__(self,idx): # 负责按索引取出某个数据，并对该数据做预处理
        root = self.filename_lists[idx]
        jnd_root = self.jnd_filename_lists[idx]
        # img, jnd = self.read_img(rootdir=root, jnd_dir=jnd_root)
        # (h, w, c) = img.shape

        while True:
            # 读取图像
            img, jnd = self.read_img(rootdir=root, jnd_dir=jnd_root)

            # 获取图像尺寸
            (h, w, c) = img.shape

            # 如果图像尺寸大于256，则跳出循环
            if h > 256 and w > 256:
                break
            
            # 否则尝试下一个文件
            idx += 1
            root = self.filename_lists[idx]
            jnd_root = self.jnd_filename_lists[idx]


        h_ran = numpy.random.randint(0, (h - 256))
        w_ran = numpy.random.randint(0, (w - 256))
        crop = img[h_ran:h_ran + 256, w_ran:w_ran + 256, :]
        jnd_crop = jnd[h_ran:h_ran + 256, w_ran:w_ran + 256, :]
        crop = torch.Tensor(crop)
        jnd_crop = torch.Tensor(jnd_crop)
        crop = crop/255
        sample = torch.cat([crop, jnd_crop], dim=2)
        sample = sample.permute(2, 0, 1)
        return sample
        

class image_(Dataset):
    def __init__(self, txt_file, root_dir):
        super(image_, self).__init__()
        f = open(txt_file, "r")
        self.list = f.readlines()
        a = len(self.list)
        for i in range(a):
            self.list[i] = self.list[i].strip().replace('\n', '')#去掉列表中每一个元素的换行符
        f.close()
        self.root_dir = root_dir

    def __len__(self):
        return len(self.list)

    def read_img(self,rootdir):
        img = cv2.imread(rootdir + "/1.jpg")
        (h, w, c) = img.shape
        h_ran = numpy.random.randint(0, (h - 256))
        w_ran = numpy.random.randint(0, (w - 256))
        crop = img[h_ran:h_ran + 256, w_ran:w_ran + 256, :] / 255
        crop = torch.Tensor(crop)
        sample = crop.permute(2, 0, 1)
        return sample

    def __getitem__(self,idx): # 负责按索引取出某个数据，并对该数据做预处理
        root = self.root_dir+ "/"+self.list[idx]
        img = self.read_img(rootdir=root)
        return img



