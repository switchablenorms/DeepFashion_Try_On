import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
import torch.nn as nn

import cv2
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F

NC = 20


def generate_discrete_label(inputs, label_nc, onehot=True, encode=True):
    pred_batch = []
    size = inputs.size()
    for input in inputs:
        input = input.view(1, label_nc, size[2], size[3])
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)

    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)
    label_map = []
    for p in pred_batch:
        p = p.view(1, 256, 192)
        label_map.append(p)
    label_map = torch.stack(label_map, 0)
    if not onehot:
        return label_map.float().cuda()
    size = label_map.size()
    oneHot_size = (size[0], label_nc, size[2], size[3])
    input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
    input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)

    return input_label
def morpho(mask,iter,bigger=True):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    new=[]
    for i in range(len(mask)):
        tem=mask[i].cpu().detach().numpy().squeeze().reshape(256,192,1)*255
        tem=tem.astype(np.uint8)
        if bigger:
            tem=cv2.dilate(tem,kernel,iterations=iter)
        else:
            tem=cv2.erode(tem,kernel,iterations=iter)
        tem=tem.astype(np.float64)
        tem=tem.reshape(1,256,192)
        new.append(tem.astype(np.float64)/255.0)
    new=np.stack(new)
    new=torch.FloatTensor(new).cuda()
    return new

def morpho_smaller(mask,iter,bigger=True):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    new=[]
    for i in range(len(mask)):
        tem=mask[i].cpu().detach().numpy().squeeze().reshape(256,192,1)*255
        tem=tem.astype(np.uint8)
        if bigger:
            tem=cv2.dilate(tem,kernel,iterations=iter)
        else:
            tem=cv2.erode(tem,kernel,iterations=iter)
        tem=tem.astype(np.float64)
        tem=tem.reshape(1,256,192)
        new.append(tem.astype(np.float64)/255.0)
    new=np.stack(new)
    new=torch.FloatTensor(new).cuda()
    return new


def encode(label_map, size):
    label_nc = 14
    oneHot_size = (size[0], label_nc, size[2], size[3])
    input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
    input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
    return input_label


class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'

    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True)

        def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake):
            return [l for (l, f) in zip((g_gan, g_gan_feat, g_vgg, d_real, d_fake), flags) if f]

        return loss_filter

    def get_G(self, in_C, out_c, n_blocks, opt, L=1, S=1):
        return networks.define_G(in_C, out_c, opt.ngf, opt.netG, L, S,
                                 opt.n_downsample_global, n_blocks, opt.n_local_enhancers,
                                 opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)

    def get_D(self, inc, opt):
        netD = networks.define_D(inc, opt.ndf, opt.n_layers_D, opt.norm, opt.no_lsgan,
                                 opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
        return netD

    def cross_entropy2d(self, input, target, weight=None, size_average=True):
        n, c, h, w = input.size()
        nt, ht, wt = target.size()

        # Handle inconsistent size between input and target
        if h != ht or w != wt:
            input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(
            input, target, weight=weight, size_average=size_average, ignore_index=250
        )

        return loss

    def ger_average_color(self, mask, arms):
        color = torch.zeros(arms.shape).cuda()
        for i in range(arms.shape[0]):
            count = len(torch.nonzero(mask[i, :, :, :]))
            if count < 10:
                color[i, 0, :, :] = 0
                color[i, 1, :, :] = 0
                color[i, 2, :, :] = 0

            else:
                color[i, 0, :, :] = arms[i, 0, :, :].sum() / count
                color[i, 1, :, :] = arms[i, 1, :, :].sum() / count
                color[i, 2, :, :] = arms[i, 2, :, :].sum() / count
        return color

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain:  # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc
        self.count = 0
        ##### define networks
        # Generator network
        netG_input_nc = input_nc
        # Main Generator
        with torch.no_grad():
            self.Unet = networks.define_UnetMask(4, self.gpu_ids).eval()
            self.G1 = networks.define_Refine(37, 14, self.gpu_ids).eval()
            self.G2 = networks.define_Refine(19+18, 1, self.gpu_ids).eval()
            self.G = networks.define_Refine(24, 3, self.gpu_ids).eval()

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.BCE = torch.nn.BCEWithLogitsLoss()

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.output_nc
            netB_input_nc = opt.output_nc * 2
            # self.D1 = self.get_D(17, opt)
            # self.D2 = self.get_D(4, opt)
            # self.D3=self.get_D(7+3,opt)
            # self.D = self.get_D(20, opt)
            # self.netB = networks.define_B(netB_input_nc, opt.output_nc, 32, 3, 3, opt.norm, gpu_ids=self.gpu_ids)

        if self.opt.verbose:
            print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.Unet, 'U', opt.which_epoch, pretrained_path)
            self.load_network(self.G1, 'G1', opt.which_epoch, pretrained_path)
            self.load_network(self.G2, 'G2', opt.which_epoch, pretrained_path)
            self.load_network(self.G, 'G', opt.which_epoch, pretrained_path)
        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss)

            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)
            self.criterionStyle = networks.StyleLoss(self.gpu_ids)
            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_fake')
            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:
                import sys
                if sys.version_info >= (3, 0):
                    finetune_list = set()
                else:
                    from sets import Set
                    finetune_list = Set()

                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():
                    if key.startswith('model' + str(opt.n_local_enhancers)):
                        params += [value]
                        finetune_list.add(key.split('.')[0])
                print(
                    '------------- Only training the local enhancer ork (for %d epochs) ------------' % opt.niter_fix_global)
                print('The layers that are finetuned are ', sorted(finetune_list))



    def encode_input(self, label_map, clothes_mask, all_clothes_label):

        size = label_map.size()
        oneHot_size = (size[0], 14, size[2], size[3])
        input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)

        masked_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        masked_label = masked_label.scatter_(1, (label_map * (1 - clothes_mask)).data.long().cuda(), 1.0)

        c_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        c_label = c_label.scatter_(1, all_clothes_label.data.long().cuda(), 1.0)

        input_label = Variable(input_label)

        return input_label, masked_label, c_label

    def encode_input_test(self, label_map, label_map_ref, real_image_ref, infer=False):

        if self.opt.label_nc == 0:
            input_label = label_map.data.cuda()
            input_label_ref = label_map_ref.data.cuda()
        else:
            # create one-hot vector for label map
            size = label_map.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
            input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
            input_label_ref = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label_ref = input_label_ref.scatter_(1, label_map_ref.data.long().cuda(), 1.0)
            if self.opt.data_type == 16:
                input_label = input_label.half()
                input_label_ref = input_label_ref.half()

        input_label = Variable(input_label, volatile=infer)
        input_label_ref = Variable(input_label_ref, volatile=infer)
        real_image_ref = Variable(real_image_ref.data.cuda())

        return input_label, input_label_ref, real_image_ref

    def discriminate(self, netD, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return netD.forward(fake_query)
        else:
            return netD.forward(input_concat)

    def gen_noise(self, shape):
        noise = np.zeros(shape, dtype=np.uint8)
        ### noise
        noise = cv2.randn(noise, 0, 255)
        noise = np.asarray(noise / 255, dtype=np.uint8)
        noise = torch.tensor(noise, dtype=torch.float32)
        return noise.cuda()
    def multi_scale_blend(self,fake_img,fake_c,mask,number=4):
        alpha=[0,0.1,0.3,0.6,0.9]
        smaller=mask
        out=0
        for i in range(1,number+1):
            bigger=smaller
            smaller=morpho(smaller,2,False)
            mid=bigger-smaller
            out+=mid*(alpha[i]*fake_c+(1-alpha[i])*fake_img)
        out+=smaller*fake_c
        out+=(1-mask)*fake_img
        return out
    def forward(self, label, pre_clothes_mask, img_fore, clothes_mask, clothes, all_clothes_label, real_image, pose,grid,mask_fore):
        # Encode Inputs
        input_label, masked_label, all_clothes_label = self.encode_input(label, clothes_mask, all_clothes_label)
        arm1_mask = torch.FloatTensor((label.cpu().numpy() == 11).astype(np.float)).cuda()
        arm2_mask = torch.FloatTensor((label.cpu().numpy() == 13).astype(np.float)).cuda()
        pre_clothes_mask=torch.FloatTensor((pre_clothes_mask.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()
        clothes = clothes * pre_clothes_mask

        shape = pre_clothes_mask.shape

        G1_in = torch.cat([pre_clothes_mask, clothes, all_clothes_label, pose, self.gen_noise(shape)], dim=1)
        arm_label = self.G1.refine(G1_in)

        arm_label = self.sigmoid(arm_label)
        CE_loss = self.cross_entropy2d(arm_label, (label * (1 - clothes_mask)).transpose(0, 1)[0].long()) * 10

        armlabel_map = generate_discrete_label(arm_label.detach(), 14, False)
        dis_label = generate_discrete_label(arm_label.detach(), 14)
        G2_in = torch.cat([pre_clothes_mask, clothes, dis_label,pose,self.gen_noise(shape)], 1)
        fake_cl = self.G2.refine(G2_in)
        fake_cl = self.sigmoid(fake_cl)
        CE_loss += self.BCE(fake_cl, clothes_mask) * 10

        fake_cl_dis = torch.FloatTensor((fake_cl.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()
        fake_cl_dis=morpho(fake_cl_dis,1,True)

        new_arm1_mask = torch.FloatTensor((armlabel_map.cpu().numpy() == 11).astype(np.float)).cuda()
        new_arm2_mask = torch.FloatTensor((armlabel_map.cpu().numpy() == 13).astype(np.float)).cuda()
        fake_cl_dis=fake_cl_dis*(1- new_arm1_mask)*(1-new_arm2_mask)
        fake_cl_dis*=mask_fore

        arm1_occ = clothes_mask * new_arm1_mask
        arm2_occ = clothes_mask * new_arm2_mask
        bigger_arm1_occ=morpho(arm1_occ,10)
        bigger_arm2_occ=morpho(arm2_occ,10  )
        arm1_full = arm1_occ + (1 - clothes_mask) * arm1_mask
        arm2_full = arm2_occ + (1 - clothes_mask) * arm2_mask
        armlabel_map *= (1 - new_arm1_mask)
        armlabel_map *= (1 - new_arm2_mask)
        armlabel_map = armlabel_map * (1 - arm1_full) + arm1_full * 11
        armlabel_map = armlabel_map * (1 - arm2_full) + arm2_full * 13
        armlabel_map*=(1-fake_cl_dis)
        dis_label=encode(armlabel_map,armlabel_map.shape)

        fake_c, warped, warped_mask,warped_grid= self.Unet(clothes, fake_cl_dis, pre_clothes_mask,grid)
        mask=fake_c[:,3,:,:]
        mask=self.sigmoid(mask)*fake_cl_dis
        fake_c = self.tanh(fake_c[:,0:3,:,:])
        fake_c=fake_c*(1-mask)+mask*warped
        skin_color = self.ger_average_color((arm1_mask + arm2_mask - arm2_mask * arm1_mask),
                                            (arm1_mask + arm2_mask - arm2_mask * arm1_mask) * real_image)
        occlude = (1 - bigger_arm1_occ * (arm2_mask + arm1_mask+clothes_mask)) * (1 - bigger_arm2_occ * (arm2_mask + arm1_mask+clothes_mask))
        img_hole_hand = img_fore * (1 - clothes_mask) * occlude * (1 - fake_cl_dis)

        G_in = torch.cat([img_hole_hand, dis_label, fake_c, skin_color, self.gen_noise(shape)], 1)
        fake_image = self.G.refine(G_in.detach())
        fake_image = self.tanh(fake_image)

        loss_D_fake = 0
        loss_D_real = 0
        loss_G_GAN = 0
        loss_G_VGG = 0

        L1_loss =0

        style_loss = L1_loss

        return [self.loss_filter(loss_G_GAN, 0, loss_G_VGG, loss_D_real, loss_D_fake), fake_image,
                clothes, arm_label
            , L1_loss, style_loss, fake_cl, CE_loss,real_image,warped_grid]

    def inference(self, label, label_ref, image_ref):

        # Encode Inputs
        image_ref = Variable(image_ref)
        input_label, input_label_ref, real_image_ref = self.encode_input_test(Variable(label), Variable(label_ref),
                                                                              image_ref, infer=True)

        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_image = self.netG.forward(input_label, input_label_ref, real_image_ref)
        else:
            fake_image = self.netG.forward(input_label, input_label_ref, real_image_ref)
        return fake_image

    def save(self, which_epoch):
        # self.save_network(self.Unet, 'U', which_epoch, self.gpu_ids)
        # self.save_network(self.G, 'G', which_epoch, self.gpu_ids)
        # self.save_network(self.G1, 'G1', which_epoch, self.gpu_ids)
        # self.save_network(self.G2, 'G2', which_epoch, self.gpu_ids)
        # # self.save_network(self.G3, 'G3', which_epoch, self.gpu_ids)
        # self.save_network(self.D, 'D', which_epoch, self.gpu_ids)
        # self.save_network(self.D1, 'D1', which_epoch, self.gpu_ids)
        # self.save_network(self.D2, 'D2', which_epoch, self.gpu_ids)
        # self.save_network(self.D3, 'D3', which_epoch, self.gpu_ids)

        pass

        # self.save_network(self.netB, 'B', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr


class InferenceModel(Pix2PixHDModel):
    def forward(self, inp):
        label = inp
        return self.inference(label)

