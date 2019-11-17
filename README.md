SSPP-DAN in Tensorflow
====

Tensorflow implementation of [SSPP-DAN: Deep Domain Adaptation Network for Face Recognition with Single Sample Per Person](https://arxiv.org/abs/1702.04069)

![Alt text](./figure/overallflow.PNG)

Prerequisites
-------------
* [Python 3.5.2](https://www.python.org/downloads/release/python-352/)
* [Tensorflow 1.12](docker pull tensorflow/tensorflow:1.12.0-gpu-py3)
* [OpenCV 4.1.1](http://opencv.org/releases.html)
* [SciPy](https://www.scipy.org/install.html)

We recommend the following instuctions.
- docker pull image (docker pull tensorflow/tensorflow:1.12.0-gpu-py3)
- in the docker container 
   apt-get update
   pip install scikit-image
   apt-get install -y libsm6 libxext6 libxrender-dev
   pip install opencv-python

Usage
-------------

First, download the [dataset](https://drive.google.com/uc?id=0ByHRRxErVc0NRjFzTXhRSUlyZlU&export=download) or the [pickle files](https://drive.google.com/uc?id=0ByHRRxErVc0NNFFINFJ2MXlvTGs&export=download
) that we have already created from our repository. After all pickle files are download, move them into the SSPP-DAN/data folder.

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
Sungeun Hong / @[csehong][wp]
[wp]: www.csehong.com


