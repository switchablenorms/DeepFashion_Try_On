## Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset, make_dataset_test
from PIL import Image
import torch
import json
import numpy as np
import os.path as osp
from PIL import ImageDraw
import ipdb

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    
        self.diction={}
        ### input A (label maps)
        if opt.isTrain or opt.use_encoded_image:
            dir_A = '_A' if self.opt.label_nc == 0 else '_label'
            self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
            self.A_paths = sorted(make_dataset(self.dir_A))
            self.AR_paths = make_dataset(self.dir_A)

        self.fine_height=256
        self.fine_width=192
        self.radius=5
        ### input A test (label maps)
        if not (opt.isTrain or opt.use_encoded_image):
            dir_A = '_A' if self.opt.label_nc == 0 else '_label'
            self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
            self.A_paths = sorted(make_dataset_test(self.dir_A))
            dir_AR = '_AR' if self.opt.label_nc == 0 else '_labelref'
            self.dir_AR = os.path.join(opt.dataroot, opt.phase + dir_AR)
            self.AR_paths = sorted(make_dataset_test(self.dir_AR))

        ### input B (real images)
        dir_B = '_B' if self.opt.label_nc == 0 else '_img'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.BR_paths = sorted(make_dataset(self.dir_B))

        self.dataset_size = len(self.A_paths)
        self.build_index(self.B_paths)

        ### input E (edge_maps)
        if opt.isTrain or opt.use_encoded_image:
            dir_E = '_edge'
            self.dir_E = os.path.join(opt.dataroot, opt.phase + dir_E)
            self.E_paths = sorted(make_dataset(self.dir_E))
            self.ER_paths = make_dataset(self.dir_E)

        ### input M (masks)
        if opt.isTrain or opt.use_encoded_image:
            dir_M = '_mask'
            self.dir_M = os.path.join(opt.dataroot, opt.phase + dir_M)
            self.M_paths = sorted(make_dataset(self.dir_M))
            self.MR_paths = make_dataset(self.dir_M)

        ### input MC(color_masks)
        if opt.isTrain or opt.use_encoded_image:
            dir_MC = '_colormask'
            self.dir_MC = os.path.join(opt.dataroot, opt.phase + dir_MC)
            self.MC_paths = sorted(make_dataset(self.dir_MC))
            self.MCR_paths = make_dataset(self.dir_MC)
        ### input C(color)
        if opt.isTrain or opt.use_encoded_image:
            dir_C = '_color'
            self.dir_C = os.path.join(opt.dataroot, opt.phase + dir_C)
            self.C_paths = sorted(make_dataset(self.dir_C))
            self.CR_paths = make_dataset(self.dir_C)
        # self.build_index(self.C_paths)

        ### input A test (label maps)
        if not (opt.isTrain or opt.use_encoded_image):
            dir_A = '_A' if self.opt.label_nc == 0 else '_label'
            self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
            self.A_paths = sorted(make_dataset_test(self.dir_A))
    def random_sample(self,item):
        name = item.split('/')[-1]
        name = name.split('-')[0]
        lst=self.diction[name]
        new_lst=[]
        for dir in lst:
            if dir != item:
                new_lst.append(dir)
        return new_lst[np.random.randint(len(new_lst))]
    def build_index(self,dirs):
        #ipdb.set_trace()
        for k,dir in enumerate(dirs):
            name=dir.split('/')[-1]
            name=name.split('-')[0]

            # print(name)
            for k,d in enumerate(dirs[max(k-20,0):k+20]):
                if name in d:
                    if name not in self.diction.keys():
                        self.diction[name]=[]
                        self.diction[name].append(d)
                    else:
                        self.diction[name].append(d)


    def __getitem__(self, index):        
        train_mask=9600
        ### input A (label maps)
        # box=[]
        # for k,x in enumerate(self.A_paths):
        #     if '2372656' in x :
        #         box.append(k)
        # index=box[np.random.randint(len(box))]
        test=index#np.random.randint(10000)
        A_path = self.A_paths[index]
        AR_path = self.AR_paths[index]
        A = Image.open(A_path).convert('L')
        AR = Image.open(AR_path).convert('L')

        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB'))
            AR_tensor = transform_A(AR.convert('RGB'))
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0
            AR_tensor = transform_A(AR) * 255.0
        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        B_path = self.B_paths[index]
        BR_path = self.BR_paths[index]
        B = Image.open(B_path).convert('RGB')
        BR = Image.open(BR_path).convert('RGB')
        transform_B = get_transform(self.opt, params)      
        B_tensor = transform_B(B)
        BR_tensor = transform_B(BR)

        ### input M (masks)
        M_path = self.M_paths[np.random.randint(12000)]
        MR_path =self.MR_paths[np.random.randint(12000)]
        M = Image.open(M_path).convert('L')
        MR = Image.open(MR_path).convert('L')
        M_tensor = transform_A(MR)

        ### input_MC (colorMasks)
        MC_path = B_path#self.MC_paths[1]
        MCR_path = B_path#self.MCR_paths[1]
        MCR = Image.open(MCR_path).convert('L')
        MC_tensor = transform_A(MCR)

        ### input_C (color)
        # print(self.C_paths)
        C_path = self.C_paths[test]
        C = Image.open(C_path).convert('RGB')
        C_tensor = transform_B(C)

        ##Edge
        E_path = self.E_paths[test]
        # print(E_path)
        E = Image.open(E_path).convert('L')
        E_tensor = transform_A(E)


        ##Pose
        pose_name =B_path.replace('.png', '_keypoints.json').replace('.jpg','_keypoints.json').replace('train_img','train_pose')
        with open(osp.join(pose_name), 'r') as f:
            pose_label = json.load(f)
            try:
                pose_data = pose_label['people'][0]['pose_keypoints']
            except IndexError:
                pose_data = [0 for i in range(54)]
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1,3))

        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        r = self.radius
        im_pose = Image.new('L', (self.fine_width, self.fine_height))
        pose_draw = ImageDraw.Draw(im_pose)
        for i in range(point_num):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i,0]
            pointy = pose_data[i,1]
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
                pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
            one_map = transform_B(one_map.convert('RGB'))
            pose_map[i] = one_map[0]
        P_tensor=pose_map
        if self.opt.isTrain:
            input_dict = { 'label': A_tensor, 'label_ref': AR_tensor, 'image': B_tensor, 'image_ref': BR_tensor, 'path': A_path, 'path_ref': AR_path,
                            'edge': E_tensor,'color': C_tensor, 'mask': M_tensor, 'colormask': MC_tensor,'pose':P_tensor
                          }
        else:
            input_dict = {'label': A_tensor, 'label_ref': AR_tensor, 'image': B_tensor, 'image_ref': BR_tensor, 'path': A_path, 'path_ref': AR_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'
