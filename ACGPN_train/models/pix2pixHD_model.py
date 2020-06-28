### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
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
import ipdb

NC=20
def generate_discrete_label(inputs, label_nc,onehot=True,encode=True):
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
def encode(label_map,size):
    label_nc=14
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
            return [l for (l,f) in zip((g_gan,g_gan_feat,g_vgg,d_real,d_fake),flags) if f]
        return loss_filter
    def get_G(self,in_C,out_c,n_blocks,opt,L=1,S=1):
        return   networks.define_G(in_C, out_c, opt.ngf, opt.netG,L,S,
                                          opt.n_downsample_global, n_blocks, opt.n_local_enhancers,
                                          opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)
    def get_D(self,inc,opt):
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
    def ger_average_color(self,mask,arms):
        color=torch.zeros(arms.shape).cuda()
        for i in range(arms.shape[0]):
            count = len(torch.nonzero(mask[i, :, :, :]))
            if count < 10:
                color[i, 0, :, :]=0
                color[i, 1, :, :]=0
                color[i, 2, :, :]=0

            else:
                color[i,0,:,:]=arms[i,0,:,:].sum()/count
                color[i,1,:,:]=arms[i,1,:,:].sum()/count
                color[i,2,:,:]=arms[i,2,:,:].sum()/count
        return color
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain: # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc
        self.count=0
        self.perm=torch.randperm(1024*4)
        ##### define networks        
        # Generator network
        netG_input_nc = input_nc        
        # Main Generator
        with torch.no_grad():
            pass
        self.Unet=networks.define_UnetMask(4,self.gpu_ids)
        self.G1 = networks.define_Refine(37,14,self.gpu_ids)
        self.G2 = networks.define_Refine(19+18,1,self.gpu_ids)
        self.G = networks.define_Refine(24,3,self.gpu_ids)
        #ipdb.set_trace()
        self.tanh=nn.Tanh()
        self.sigmoid=nn.Sigmoid()
        self.BCE=torch.nn.BCEWithLogitsLoss()

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.output_nc
            netB_input_nc = opt.output_nc * 2
            self.D1=self.get_D(34+14+3,opt)
            self.D2=self.get_D(20+18,opt)
            self.D=self.get_D(27,opt)
            self. D3=self.get_D(7,opt)
            #self.netB = networks.define_B(netB_input_nc, opt.output_nc, 32, 3, 3, opt.norm, gpu_ids=self.gpu_ids)        
            
        if self.opt.verbose:
                print('---------- Networks initialized -------------')

        # load networks
        # if not self.isTrain or opt.continue_train or opt.load_pretrain:
        #     pretrained_path = '' if not self.isTrain else opt.load_pretrain
        #     self.load_network(self.Unet, 'U', opt.which_epoch, pretrained_path)
        #     self.load_network(self.G1, 'G1', opt.which_epoch, pretrained_path)
        #     self.load_network(self.G2, 'G2', opt.which_epoch, pretrained_path)
        #     self.load_network(self.G, 'G', opt.which_epoch, pretrained_path)
        #     self.load_network(self.D, 'D', opt.which_epoch, pretrained_path)
        #     self.load_network(self.D1, 'D1', opt.which_epoch, pretrained_path)
        #     self.load_network(self.D2, 'D2', opt.which_epoch, pretrained_path)
        #     self.load_network(self.D3, 'D3', opt.which_epoch, pretrained_path)
        #     self.load_network(self.optimizer_G, 'OG', opt.which_epoch, pretrained_path)
        #     self.load_network(self.optimizer_D, 'OD', opt.which_epoch, pretrained_path)



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
            self.criterionStyle=networks.StyleLoss(self.gpu_ids)
            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN','G_GAN_Feat','G_VGG','D_real','D_fake')
            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:                
                import sys
                if sys.version_info >= (3,0):
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
                print('------------- Only training the local enhancer ork (for %d epochs) ------------' % opt.niter_fix_global)
                print('The layers that are finetuned are ', sorted(finetune_list))                         
            else:
                params =list(self.Unet.parameters())+list(self.G.parameters())+list(self.G1.parameters())+list(self.G2.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=0.0002, betas=(opt.beta1, 0.999))

            # optimizer D                        
            params =list(self.D3.parameters())+list(self.D.parameters())+list(self.D2.parameters())+list(self.D1.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=0.0002, betas=(opt.beta1, 0.999))

            # load networks
            if not self.isTrain or opt.continue_train or opt.load_pretrain:
                pretrained_path = '' if not self.isTrain else opt.load_pretrain
                self.load_network(self.Unet, 'U', opt.which_epoch, pretrained_path)
                self.load_network(self.G1, 'G1', opt.which_epoch, pretrained_path)
                self.load_network(self.G2, 'G2', opt.which_epoch, pretrained_path)
                self.load_network(self.G, 'G', opt.which_epoch, pretrained_path)
                self.load_network(self.D, 'D', opt.which_epoch, pretrained_path)
                self.load_network(self.D1, 'D1', opt.which_epoch, pretrained_path)
                self.load_network(self.D2, 'D2', opt.which_epoch, pretrained_path)
                self.load_network(self.D3, 'D3', opt.which_epoch, pretrained_path)
                #self.load_network(self.optimizer_G, 'OG', opt.which_epoch, pretrained_path)
                #self.load_network(self.optimizer_D, 'OD', opt.which_epoch, pretrained_path)
   
            # optimizer G + B                        
            #params = list(self.netG.parameters()) + list(self.netB.parameters())     
            #self.optimizer_GB = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
    
    def encode_input(self,label_map, clothes_mask,all_clothes_label):
        size = label_map.size()
        oneHot_size = (size[0], 14, size[2], size[3])
        input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)

        masked_label= torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        masked_label=masked_label.scatter_(1,(label_map*(1-clothes_mask)).data.long().cuda(), 1.0)

        c_label=torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        c_label=c_label.scatter_(1,all_clothes_label.data.long().cuda(),1.0)
 
        input_label = Variable(input_label)

        return input_label,masked_label,c_label

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

    def discriminate(self,netD ,input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return netD.forward(fake_query)
        else:
            return netD.forward(input_concat)
    def gen_noise(self,shape):
        noise = np.zeros(shape, dtype=np.uint8)
        ### noise
        noise = cv2.randn(noise, 0, 255)
        noise = np.asarray(noise / 255, dtype=np.uint8)
        noise = torch.tensor(noise, dtype=torch.float32)
        return noise.cuda()
    #data['label'],data['edge'],img_fore.cuda()),Variable(mask_clothes, ,Variable(data['color'].cuda()),Variable(all_clothes_label.cuda()))
    
    def forward(self,label,pre_clothes_mask,img_fore,clothes_mask,clothes,all_clothes_label,real_image,pose,mask):
        # Encode Inputs
        #ipdb.set_trace()
        input_label,masked_label,all_clothes_label= self.encode_input(label,clothes_mask,all_clothes_label)
        #ipdb.set_trace()
        arm1_mask=torch.FloatTensor((label.cpu().numpy()==11).astype(np.float)).cuda()
        arm2_mask=torch.FloatTensor((label.cpu().numpy()==13).astype(np.float)).cuda()
        pre_clothes_mask=torch.FloatTensor((pre_clothes_mask.detach().cpu().numpy() > 0.5).astype(np.float)).cuda()
        clothes=clothes*pre_clothes_mask

        #clothes_mask -> target
        #pre_clothes_mask -> source

        # fake_image,warped,warped_mask=self.Unet(clothes,clothes_mask,pre_clothes_mask)
        # real_image=real_image * clothes_mask+(1-clothes_mask)*-1
        shape=pre_clothes_mask.shape


        G1_in=torch.cat([pre_clothes_mask,clothes,all_clothes_label,pose,self.gen_noise(shape)],dim=1)
        arm_label=self.G1.refine(G1_in)
        arm_label=self.sigmoid(arm_label)
        CE_loss = self.cross_entropy2d(arm_label, (label * (1 - clothes_mask)).transpose(0, 1)[0].long())*10


        armlabel_map=generate_discrete_label(arm_label.detach(),14,False)
        dis_label=generate_discrete_label(arm_label.detach(),14)

        G2_in=torch.cat([pre_clothes_mask,clothes,masked_label,pose,self.gen_noise(shape)],1)
        fake_cl=self.G2.refine(G2_in)
        fake_cl=self.sigmoid(fake_cl)
        CE_loss += self.BCE(fake_cl, clothes_mask)*10
        
        #ipdb.set_trace()
        fake_cl_dis=torch.FloatTensor((fake_cl.detach().cpu().numpy()>0.5).astype(np.float)).cuda()
        new_arm1_mask=torch.FloatTensor((armlabel_map.cpu().numpy()==11).astype(np.float)).cuda()
        new_arm2_mask=torch.FloatTensor((armlabel_map.cpu().numpy()==13).astype(np.float)).cuda()
        arm1_occ=clothes_mask*new_arm1_mask
        arm2_occ=clothes_mask*new_arm2_mask
        arm1_full=arm1_occ+(1-clothes_mask)*arm1_mask
        arm2_full=arm2_occ+(1-clothes_mask)*arm2_mask
        armlabel_map*=(1-new_arm1_mask)
        armlabel_map*=(1-new_arm2_mask)
        armlabel_map=armlabel_map*(1-arm1_full)+arm1_full*11
        armlabel_map=armlabel_map*(1-arm2_full)+arm2_full*13


        ## construct full label map
        armlabel_map=armlabel_map*(1-fake_cl_dis)+fake_cl_dis*4


 
        fake_c, warped, warped_mask,rx,ry,cx,cy,rg,cg = self.Unet(clothes, clothes_mask,pre_clothes_mask)
        #ipdb.set_trace()
        composition_mask = fake_c[:, 3, :, :]
        fake_c=fake_c[:,0:3,:,:]
        fake_c=self.tanh(fake_c)
        composition_mask=self.sigmoid(composition_mask)

        skin_color=self.ger_average_color((arm1_mask+arm2_mask-arm2_mask*arm1_mask),(arm1_mask+arm2_mask-arm2_mask*arm1_mask)*real_image)

        img_hole_hand=img_fore*(1-clothes_mask)*(1-arm1_mask)*(1-arm2_mask)+img_fore*arm1_mask*(1-mask)+img_fore*arm2_mask*(1-mask)

        G_in=torch.cat([img_hole_hand,masked_label,real_image*clothes_mask,skin_color,self.gen_noise(shape)],1)
        fake_image=self.G.refine(G_in.detach())
        fake_image=self.tanh(fake_image)
        ## THE POOL TO SAVE IMAGES\
        ##

        input_pool=[G1_in,G2_in,G_in,torch.cat([clothes_mask,clothes],1)]        ##fake_cl_dis to replace
        #ipdb.set_trace()
        real_pool=[masked_label,clothes_mask,real_image,real_image*clothes_mask]
        fake_pool=[arm_label,fake_cl,fake_image,fake_c]
        D_pool=[self.D1,self.D2,self.D,self.D3]
        pool_lenth=len(fake_pool)
        loss_D_fake=0
        loss_D_real=0
        loss_G_GAN=0
        loss_G_GAN_Feat = 0

        for iter_p in range(pool_lenth):

            # Fake Detection and Loss
            pred_fake_pool = self.discriminate(D_pool[iter_p],input_pool[iter_p].detach(), fake_pool[iter_p], use_pool=True)
            loss_D_fake += self.criterionGAN(pred_fake_pool, False)
            # Real Detection and Loss
            pred_real = self.discriminate(D_pool[iter_p],input_pool[iter_p].detach(), real_pool[iter_p])
            loss_D_real += self.criterionGAN(pred_real, True)

            # GAN loss (Fake Passability Loss)
            pred_fake = D_pool[iter_p].forward(torch.cat((input_pool[iter_p].detach(), fake_pool[iter_p]), dim=1))
            loss_G_GAN += self.criterionGAN(pred_fake, True)
            if iter_p <2:
                continue
            # # GAN feature matching loss
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat

        #ipdb.set_trace()
        comp_fake_c=fake_c.detach()*(1-composition_mask).unsqueeze(1)+(composition_mask.unsqueeze(1))*warped.detach()

        # VGG feature matching loss
        loss_G_VGG = 0
        loss_G_VGG +=self.criterionVGG.warp(warped,real_image*clothes_mask)+ self.criterionVGG.warp(comp_fake_c, real_image*clothes_mask) * 10
        loss_G_VGG += self.criterionVGG.warp(fake_c, real_image*clothes_mask) *20
        loss_G_VGG += self.criterionVGG(fake_image, real_image) *10

        L1_loss=self.criterionFeat(fake_image , real_image )
        #
        L1_loss+=self.criterionFeat(warped_mask,clothes_mask)+self.criterionFeat(warped,real_image*clothes_mask)
        L1_loss+=self.criterionFeat(fake_c,real_image*clothes_mask)*0.2
        L1_loss+=self.criterionFeat(comp_fake_c,real_image*clothes_mask)*10
        L1_loss+=self.criterionFeat(composition_mask,clothes_mask)

        #
        # style_loss=self.criterionStyle(fake_image, real_image)*200

        # loss_G_GAN_Feat=L1_loss
        style_loss=L1_loss
        # Only return the fake_B image if necessary to save BW
        return [ self.loss_filter( loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake ),fake_c,comp_fake_c,dis_label
       ,L1_loss,style_loss,fake_cl,warped,clothes,CE_loss,rx*0.1,ry*0.1,cx*0.1,cy*0.1,rg*0.1,cg*0.1]

    def inference(self, label, label_ref, image_ref):

        # Encode Inputs
        image_ref = Variable(image_ref)
        input_label, input_label_ref, real_image_ref = self.encode_input_test(Variable(label), Variable(label_ref), image_ref, infer=True)

        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_image = self.netG.forward(input_label, input_label_ref, real_image_ref)
        else:
            fake_image = self.netG.forward(input_label, input_label_ref, real_image_ref)
        return fake_image

    def save(self, which_epoch):
        self.save_network(self.Unet, 'U', which_epoch, self.gpu_ids)
        self.save_network(self.G,'G',which_epoch, self.gpu_ids)
        self.save_network(self.G1,'G1',which_epoch, self.gpu_ids)
        self.save_network(self.G2, 'G2', which_epoch, self.gpu_ids)
        self.save_network(self.D, 'D', which_epoch, self.gpu_ids)
        self.save_network(self.D1, 'D1', which_epoch, self.gpu_ids)
        self.save_network(self.D2, 'D2', which_epoch, self.gpu_ids)
        self.save_network(self.D3, 'D3', which_epoch, self.gpu_ids)
        self.save_network(self.optimizer_G, 'OG', which_epoch, self.gpu_ids)
        self.save_network(self.optimizer_D, 'OD', which_epoch, self.gpu_ids)

        pass

        #self.save_network(self.netB, 'B', which_epoch, self.gpu_ids)
      
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

        
