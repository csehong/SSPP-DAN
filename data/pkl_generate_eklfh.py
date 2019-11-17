import cv2
import os                       # help with file handling
import pickle
import numpy as np
import skimage.io
from io import BytesIO



src_protocol = 'Front_3D' #Front   Front_3D   Front_Semi   Front_3D_Semi
tgt_scenario = 2 # 1 or 2



root_dir = '/workspace/data/'
input_dir_src = root_dir + "EK-LFH/Webcam/" + src_protocol
input_dir_tgt = root_dir + 'EK-LFH/Surveillance'
outdir_dir = root_dir + "eklfh_pkl/"
if not(os.path.isdir(outdir_dir)):
        os.makedirs(os.path.join(outdir_dir))

# Source Train
dict_src_train = {}
cnt = 0
for root, dirs, files in os.walk(input_dir_src):
    for fname in files:
        img_path = os.path.join(root, fname)
        _, ext = os.path.splitext(img_path)

        if ext not in ['.jpg', '.JPG', '.png', '.PNG']:
            continue
        img  = open(img_path, 'rb').read()
        dict_src_train[fname] = img
        cnt +=1

dict_path = {'Front': 'train_ori',
             'Front_3D': 'train_ori_3D',
             'Front_Semi': 'train_ori_semi',
             'Front_3D_Semi': 'train_ori_3D_semi'}

pklName_src = outdir_dir + 'eklfh_src_' + dict_path[src_protocol] + '.pkl'

if os.path.exists(pklName_src) != True:
    with open(pklName_src, 'wb') as f:
        pickle.dump(dict_src_train, f)

print("Total Sample (Src): ", cnt)



# Target Train/Test
dict_tgt_train = {}
dict_tgt_test = {}

cnt_train = 0
cnt_test = 0
for root, dirs, files in os.walk(input_dir_tgt):
    for fname in files:
        img_path = os.path.join(root, fname)
        _, ext = os.path.splitext(img_path)

        if ext not in ['.jpg', '.JPG', '.png', '.PNG']:
            continue
        subject_id = int((fname.split('_')[1]))

        img  = open(img_path, 'rb').read()
        # scenario 1: split dataset according to set_id
        if tgt_scenario == 1:
            set_id = int(fname.split('_')[0][3])

            if set_id in [1, 2, 3, 4, 6, 7]:
                dict_tgt_train[fname] = img
                cnt_train += 1
            elif set_id in [5]:
                dict_tgt_test[fname] = img
                cnt_test += 1
            else:
                NotImplementedError('Unvalid Set ID')
        # scenario 2: split dataset according to subject_id
        elif tgt_scenario == 2:
            max_id_train = 20

            if subject_id <=max_id_train:
                dict_tgt_train[fname] = img
                cnt_train +=1
            elif subject_id >max_id_train:
                dict_tgt_test[fname] = img
                cnt_test += 1
            else:
                NotImplementedError('Unvalid Subject ID')
        else:
            NotImplementedError('Unvalid Scenario')

pklName_tgt_train = outdir_dir + 'eklfh_s' + str(tgt_scenario) + '_tgt_train.pkl'
pklName_tgt_test = outdir_dir + 'eklfh_s' + str(tgt_scenario) + '_tgt_test.pkl'

if os.path.exists(pklName_tgt_train) != True:
    with open(pklName_tgt_train, 'wb') as f:
        pickle.dump(dict_tgt_train, f)

if os.path.exists(pklName_tgt_test) != True:
    with open(pklName_tgt_test, 'wb') as f:
        pickle.dump(dict_tgt_test, f)

print("Total Sample (Tgt_Train): ", cnt_train)
print("Total Sample (Tgt_Test): ", cnt_test)

