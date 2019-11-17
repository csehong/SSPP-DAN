import os
import pickle
import random
import numpy as np
import io
from io import BytesIO
import skimage.io
import util.img_proc as img_proc
import threading
import cv2



# Class for Dataset
class DataSet:
    def __init__(self, data_pkl_path):
        try:
            with open(data_pkl_path, 'rb') as f:
                self.data = pickle.load(f)
        except UnicodeDecodeError as e:
            with open(data_pkl_path, 'rb') as f:
                self.data = pickle.load(f, encoding='latin1')
        except Exception as e:
            print('Unable to load data ', data_pkl_path, ':', e)
            raise
        except IOError:
            print("File not exist: %s" % (data_pkl_path))
            exit(-1)


        f.close()
        self.list = list(self.data.keys())

    def shuffle(self):
        random.shuffle(self.list)

    def get_num_set(self):
        return len(self.list)


# Class for Managing Dataset
class Manager:
    def __init__(self, dataset_path, dataset_name, exp_mode):
        self.dataset_name = dataset_name
        self.exp_mode = exp_mode #lower, lower_3D, dom, dom_3D, dom_3D_cycle, semi, semi_3D, semi_3D_cycle, upper

        if exp_mode in ["lower", "dom", "upper"]:
            pkl_name_src_train = "_src_train_ori.pkl"
        elif exp_mode in ["lower_3D", "dom_3D", "upper_3D"]:
            pkl_name_src_train = "_src_train_ori_3D.pkl"
        elif exp_mode in ["semi"]:
            pkl_name_src_train = "_src_train_ori_semi.pkl"
        elif exp_mode in ["semi_3D"]:
            pkl_name_src_train = "_src_train_ori_3D_semi.pkl"
        else:
            NotImplementedError('exp mode error')


        self.source_train_set = DataSet(os.path.join(dataset_path, dataset_name.split('_')[0] + "_pkl", dataset_name.split('_')[0]  + pkl_name_src_train))
        self.target_train_set = DataSet(os.path.join(dataset_path, dataset_name.split('_')[0] + "_pkl", dataset_name + '_tgt_train.pkl'))
        self.target_test_set = DataSet(os.path.join(dataset_path, dataset_name.split('_')[0] + "_pkl",  dataset_name + '_tgt_test.pkl'))
        print(self.dataset_name)
        if "eklfh" in self.dataset_name:
            self.num_class = 30
        elif "scface" in self.dataset_name:
            self.num_class = 130
        else:
            raise Exception('not defined dataset!')
        print ("Source_train: %d, Target_train: %d, Target_test: %d" % (self.source_train_set.get_num_set(), self.target_train_set.get_num_set(), self.target_test_set.get_num_set()))

    def one_hot(self, dataset_name, fname):
        if dataset_name.split('_')[0] == 'eklfh':
            pid = int(fname.split('_')[1])
        elif dataset_name.split('_')[0] == 'scface':
            pid = int(fname.split('_')[0])
        else:
            print("dataset name error")
            exit(0)

        #print(fname, pid)
        v = np.zeros(self.num_class)
        v[pid-1] = 1
        return v

    def train_jitter(self, img, w=None, h=None, f=None, s=None):
        if w is None:
            w = random.randint(224, 250)
        if h is None:
            h = w
        if f is None:
            f = random.choice([True, False])
        if s is None:
            s = random.choice([False, 0, 1])
        img = img_proc.resize(img, (w,h))
        img = img_proc.crop_center(img, (224, 224))
        if s is not False:
            img = cv2.GaussianBlur(img, (11, 11), s)
        if f:
            img = img_proc.flip(img)
        return img

    def get_random_n_selection(self, n, li):
        iter_list = list(range(len(li)))
        random.shuffle(iter_list)
        i = 0
        while True:
            res = []
            for _ in range(n):
                res.append(iter_list[i])
                i += 1
                if i >= len(iter_list):
                    i = 0
                    random.shuffle(iter_list)
            yield res

    def eval_batch_generator(self, num_batch):
        x_tt = self.target_test_set.list
        x_tt_data = self.target_test_set.data
        num_test = len(x_tt)
        i = 0
        while True:
            if i >= num_test:
                break
            batch_x = []
            batch_y = []
            wl_idx = []
            dom_one_hot = []
            for _ in range(num_batch):
                im = skimage.io.imread(BytesIO(x_tt_data[x_tt[i]]))
                #im = skimage.io.imread(x_tt_data[x_tt[i]])
                batch_x.append(img_proc.resize(im, (224, 224)))
                dom_one_hot.append(np.array([0, 1]))
                batch_y.append(self.one_hot(self.dataset_name, x_tt[i]))
                wl_idx.append(len(batch_x) - 1)
                i += 1
                if i >= num_test:
                    break
            yield np.array(batch_x), np.array(batch_y), np.array(wl_idx), np.array(dom_one_hot), (num_test - i)

    def batch_generator(self, num_batch, type='train'):
        if type == 'train':
            x_st = self.source_train_set.list
            x_tt = self.target_train_set.list
            x_st_data = self.source_train_set.data
            x_tt_data = self.target_train_set.data
            sel_st = self.get_random_n_selection(num_batch, x_st)
        elif type == 'test':
            x_tt = self.target_test_set.list
            x_tt_data = self.target_test_set.data
        sel_tt = self.get_random_n_selection(num_batch, x_tt)
        while True:
            batch_x = []
            batch_y = []
            wl_idx = []
            dom_one_hot = []
            ran = sel_tt.__next__()
            for i in ran:
                if type != 'test':
                    f = random.choice([True, False])
                else:
                    f = False
                im = skimage.io.imread(BytesIO(x_tt_data[x_tt[i]]))
                batch_x.append(self.train_jitter(im, f=f, s=False))
                dom_one_hot.append(np.array([0, 1]))

                if type == 'test':
                    batch_y.append(self.one_hot(self.dataset_name, x_tt[i]))
                    wl_idx.append(len(batch_x) - 1)

                if self.exp_mode == 'upper' and type == 'train':
                    batch_y.append(self.one_hot(self.dataset_name, x_tt[i]))
                    wl_idx.append(len(batch_x) - 1)

            if type == 'train':
                ran = sel_st.__next__()
                for i in ran:
                    f = random.choice([True, False])
                    im = skimage.io.imread(BytesIO(x_st_data[x_st[i]]))
                    #im = skimage.io.imread(x_st_data[x_st[i]])
                    batch_x.append(self.train_jitter(im, f=f))
                    batch_y.append(self.one_hot(self.dataset_name, x_st[i]))

                    # for semi-supervised learning
                    if 'Set' in x_st[i] or 'cam' in x_st[i]:
                        dom_one_hot.append(np.array([0, 1]))
                    else:
                        dom_one_hot.append(np.array([1, 0]))

                    wl_idx.append(len(batch_x) - 1)
            # print(np.array(dom_one_hot))
            yield np.array(batch_x), np.array(batch_y), np.array(wl_idx), np.array(dom_one_hot)

    def get_batch(self, b, batch_list):
        x, y, idx, d = b[0].__next__()
        batch_list[0] = x
        batch_list[1] = y
        batch_list[2] = idx
        batch_list[3] = d

    def batch_generator_thread(self, batch_num, type='train'):
        b = [self.batch_generator(batch_num, type)]
        load_fin = [False]
        batch_list = [None, None, None, None]
        th = threading.Thread(target=self.get_batch, args=(b, batch_list))
        th.start()
        while True:
            th.join()
            res = batch_list[:]
            th = threading.Thread(target=self.get_batch, args=(b, batch_list))
            th.start()
            yield res[0], res[1], res[2], res[3]

