import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm 
import json
import h5py

def get_segmentation_classes(root):
    catfile = os.path.join(root, 'synsetoffset2category.txt')
    cat = {}
    meta = {}

    with open(catfile, 'r') as f:
        for line in f:
            ls = line.strip().split()
            cat[ls[0]] = ls[1]

    for item in cat:
        dir_seg = os.path.join(root, cat[item], 'points_label')
        dir_point = os.path.join(root, cat[item], 'points')
        fns = sorted(os.listdir(dir_point))
        meta[item] = []
        for fn in fns:
            token = (os.path.splitext(os.path.basename(fn))[0])
            meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg')))
    
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'w') as f:
        for item in cat:
            datapath = []
            num_seg_classes = 0
            for fn in meta[item]:
                datapath.append((item, fn[0], fn[1]))

            for i in tqdm(range(len(datapath))):
                l = len(np.unique(np.loadtxt(datapath[i][-1]).astype(np.uint8)))
                if l > num_seg_classes:
                    num_seg_classes = l

            print("category {} num segmentation classes {}".format(item, num_seg_classes))
            f.write("{}\t{}\n".format(item, num_seg_classes))

# def gen_modelnet_id(root):
#     classes = []
#     with open(os.path.join(root, 'train.txt'), 'r') as f:
#         for line in f:
#             classes.append(line.strip().split('/')[0])
#     classes = np.unique(classes)
#     with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'w') as f:
#         for i in range(len(classes)):
#             f.write('{}\t{}\n'.format(classes[i], i))

class ShapeNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=1024,
                 classification=False,
                 class_choice=None,
                 split='train',
                 data_augmentation=True,
                 preprocessing=False):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        self.classification = classification
        self.seg_classes = {}
        self.preprocessing = preprocessing
        
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        #print(self.cat)
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}

        self.meta = {}
        splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        #from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'points', uuid+'.pts'),
                                        os.path.join(self.root, category, 'points_label', uuid+'.seg')))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        # print(self.classes)
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.seg_classes[ls[0]] = int(ls[1])
        self.num_seg_classes = self.seg_classes[list(self.cat.keys())[0]]
        # print(self.seg_classes, self.num_seg_classes)

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)
        #print(point_set.shape, seg.shape)

        try:
            # resample
            choice = np.random.choice(len(seg), self.npoints, replace=False)
            point_set = point_set[choice, :]
        except:
            # shuffle
            choice = np.random.choice(len(seg), self.npoints, replace=True)
            point_set = point_set[choice, :]
        # choice = np.random.choice(len(seg), self.npoints, replace=True)
        # point_set = point_set[choice, :]

        # maybe ShapeNet requires center and scale?
        if self.preprocessing:
            point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
            dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
            point_set = point_set / dist #scale

        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

        seg = seg[choice]
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        if self.classification:
            return point_set, cls
        else:
            return point_set, seg

    def __len__(self):
        return len(self.datapath)

class ModelNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 class_num=40,
                 npoints=1024,
                 split='train',
                 data_augmentation=True,
                 preprocessing=False):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.preprocessing = preprocessing
        if class_num == 40:
            self.classes = [line.rstrip() for line in open(os.path.join(root, 'ModelNet40_train', 'shape_names.txt'))]
            h5_chunk_menu = [line.rstrip() for line in open(os.path.join(root, 'ModelNet40_'+split, split+'_files.txt'))]
            for idx, h5_chunk in enumerate(h5_chunk_menu):
                h5_filename = os.path.join(root, 'ModelNet40_'+split, h5_chunk)
                f = h5py.File(h5_filename)
                data = f['data'][:]
                label = f['label'][:]
                if idx == 0:
                    DATA = data
                    LABEL = label
                else:
                    DATA = np.concatenate((DATA, data), axis=0)
                    LABEL = np.concatenate((LABEL, label), axis=0)
        elif class_num == 10:
            self.classes = ['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair','cone']
            h5_filename = os.path.join(root, 'ModelNet10_'+split, 'modelnet10_'+split+'.h5')
            f = h5py.File(h5_filename)
            DATA = f['data'][:]
            LABEL = f['label'][:]
        else:
            assert False

        self.fns = list(DATA)
        self.cat = list(LABEL[:,0])

    def __getitem__(self, index):
        pts = self.fns[index]
        cls = self.cat[index]

        try:
            # resample
            choice = np.random.choice(len(pts), self.npoints, replace=False)
            point_set = pts[choice, :]
        except:
            # shuffle
            choice = np.random.permutation(len(pts))
            point_set = pts[choice, :]

        ## Fuck off, we don't require center and scale, as same as TensorFlow implementation.
        if self.preprocessing:
            point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
            dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
            point_set = point_set / dist  # scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set.astype(np.float32))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        return point_set, cls


    def __len__(self):
        return len(self.fns)

if __name__ == '__main__':
    dataset = sys.argv[1]
    datapath = sys.argv[2]

    if dataset == 'shapenet':
        d = ShapeNetDataset(root = datapath, class_choice = ['Chair'])
        print(len(d))
        ps, seg = d[0]
        print(ps.size(), ps.type(), seg.size(),seg.type())

        d = ShapeNetDataset(root = datapath, classification = True)
        print(len(d))
        ps, cls = d[0]
        print(ps.size(), ps.type(), cls.size(),cls.type())
        # get_segmentation_classes(datapath)

    if dataset == 'modelnet':
        d = ModelNetDataset(root=datapath)
        print(len(d))
        print(d[0])

