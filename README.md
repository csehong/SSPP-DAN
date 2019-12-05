SSPP-DAN-TensorFlow
====

Tensorflow implementation of [SSPP-DAN: Deep Domain Adaptation Network for Face Recognition with Single Sample Per Person](https://arxiv.org/abs/1702.04069)

![Alt text](./figure/overallflow.PNG)

Prerequisites
-------------
* [Python 3.5.2](https://www.python.org/downloads/release/python-352/)
* [Tensorflow 1.12.0-gpu-py3](https://hub.docker.com/r/tensorflow/tensorflow/)
* [OpenCV 4.1.1](http://opencv.org/releases.html)
* [SciPy](https://www.scipy.org/install.html)

We recommend the following instuctions.
- Pull docker image (docker pull tensorflow/tensorflow:1.12.0-gpu-py3)
- In the docker container 
   apt-get update
   pip install scikit-image
   apt-get install -y libsm6 libxext6 libxrender-dev
   pip install opencv-python

Usage
-------------

First, download the [dataset](https://drive.google.com/open?id=1PFh3s8WL6_tmMe-oNXM73526ngXQ51TD) or the [pickle files](https://drive.google.com/open?id=1yqFCnPi8u-bEugnLBITkCIOnThKdHjIg) that we already generated. After all pickle files are download, move them into the 'SSPP-DAN/data/eklfh_pkl' folder.

Directory Tree
```
|-- DAN.py
|-- README.md
|-- data
|   |-- EK-LFH
|   |-- SCface
|   |-- __init__.py
|   |-- data_manager.py
|   |-- eklfh_pkl
|   |   |-- eklfh_s1_tgt_test.pkl
|   |   |-- eklfh_s1_tgt_train.pkl
|   |   |-- eklfh_s2_tgt_test.pkl
|   |   |-- eklfh_s2_tgt_train.pkl
|   |   |-- eklfh_src_train_ori.pkl
|   |   |-- eklfh_src_train_ori_3D.pkl
|   |   |-- eklfh_src_train_ori_3D_semi.pkl
|   |   |-- eklfh_src_train_ori_semi.pkl
|   |-- pkl_generate_eklfh.py
|   |-- pkl_generate_scface.py
|-- pretrained
|   |-- VGG_Face.py
|   |-- __init__.py
|   |-- get_vggface.sh
|-- test_model.py
|-- train_model.py
|-- util
    |-- Logger.py
    |-- OPTS.py
    |-- PyMatData.py
    |-- __init__.py
    |-- flip_gradient.py
    |-- img_proc.py
```

Then run get_vggface.sh in the SSPP-DAN/pretrained folder to use the pre-trained VGG-Face model.

To train a model with downloaded dataset:
```
$ python train_model.py --dataset='eklfh_s1' --exp_mode='dom_3D' 
```

To test with an existing model:
```
$ python test_model.py --dataset='eklfh_s1' --exp_mode='dom_3D'  --summaries_dir 'exp_eklfh_s1/tuning/exp_2_dom__batch_64__steps_10000__lr_2e-05__embfc7__dr_0.3__ft_fc7' 
```

Results
-------------
Facial feature space (left) and its embedding space after applying DA (right). The subscript “s” and “t” in the
legend refer to the source and target domains, respectively.

![Alt text](./figure/DAN.PNG)




Author
------------
Sungeun Hong 
e: csehong@gmail.com
w: www.csehong.com


