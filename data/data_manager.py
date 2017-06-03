import os
import pickle
import random
import numpy as np
from StringIO import StringIO
import skimage.io
import util.img_proc as img_proc
import threading
import cv2



# Class for Dataset
class DataSet:
    def __init__(self, data_pkl_path):
        try:
            f = open(data_pkl_path, 'rb')
        except IOError:
            print ("File not exist: %s"%(data_pkl_path))
            exit(-1)
        self.data = pickle.load(f)
        f.close()
        self.list = self.data.keys()

    def shuffle(self):
        random.shuffle(self.list)

    def get_num_set(self):
        return len(self.list)


# Class for Managing Dataset
class Manager:
    def __init__(self, data_path, num_class):
        self.source_train_set = DataSet(os.path.join(data_path, 'src_train.pkl'))
        self.target_train_set = DataSet(os.path.join(data_path, 'target_train.pkl'))
        self.target_test_set = DataSet(os.path.join(data_path, 'target_test.pkl'))
        self.num_class = num_class
        print ("Source_train: %d s, Target_train: %d, Target_test: %d" % (self.source_train_set.get_num_set(), self.target_train_set.get_num_set(), self.target_test_set.get_num_set()))

    def one_hot(self, fname):
        pid = int(fname.split('_')[1])
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
            for _ in xrange(n):
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
                im = skimage.io.imread(StringIO(x_tt_data[x_tt[i]]))
                batch_x.append(img_proc.resize(im, (224, 224)))
                dom_one_hot.append(np.array([0, 1]))
                batch_y.append(self.one_hot(x_tt[i]))
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
            ran = sel_tt.next()
            for i in ran:
                if type != 'test':
                    f = random.choice([True, False])
                else:
                    f = False
                im = skimage.io.imread(StringIO(x_tt_data[x_tt[i]]))
                batch_x.append(self.train_jitter(im, f=f, s=False))
                dom_one_hot.append(np.array([0, 1]))
                if type == 'test':
                    batch_y.append(self.one_hot(x_tt[i]))
                    wl_idx.append(len(batch_x) - 1)
            if type == 'train':
                ran = sel_st.next()
                for i in ran:
                    f = random.choice([True, False])
                    im = skimage.io.imread(StringIO(x_st_data[x_st[i]]))
                    batch_x.append(self.train_jitter(im, f=f))
                    batch_y.append(self.one_hot(x_st[i]))
                    if x_st[i].startswith('Set'):
                        dom_one_hot.append(np.array([0, 1]))
                    else:
                        dom_one_hot.append(np.array([1, 0]))
                    wl_idx.append(len(batch_x) - 1)
            yield np.array(batch_x), np.array(batch_y), np.array(wl_idx), np.array(dom_one_hot)

    def get_batch(self, b, batch_list):
        x, y, idx, d = b[0].next()
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

