SSPP-DAN in Tensorflow
====

Tensorflow implementation of [SSPP-DAN: Deep Domain Adaptation Network for Face Recognition with Single Sample Per Person](https://arxiv.org/abs/1702.04069)

![Alt text](./figure/overallflow.PNG)

Prerequisites
-------------
* [Python 2.7](https://www.python.org/downloads/)
* [Tensorflow 0.12](https://www.tensorflow.org/versions/r0.12/)
* [OpenCV 2.4.9](http://opencv.org/releases.html)
* [NumPy](http://www.numpy.org/)
* [SciPy](https://www.scipy.org/install.html)

Usage
-------------

First, download the [dataset](https://drive.google.com/uc?id=0ByHRRxErVc0NRjFzTXhRSUlyZlU&export=download) or the [pickle files](https://drive.google.com/uc?id=0ByHRRxErVc0NNFFINFJ2MXlvTGs&export=download
) that we have already created from our repository. After all pickle files are download, move them into the SSPP-DAN/data folder.

Then run get_vggface.sh in the SSPP-DAN/pretrained folder to use the pre-trained VGG-Face model.

To train a model with downloaded dataset:
```
$ python train_model.py --learning_rate=1e-5 --batch_size=50 --save_step=100
```

To test with an existing model:
```
$ python test_model.py --summaries_dir 'expr/F3D_30_60_FC6_FC6' --test_batch_size=50
```

Results
-------------
Facial feature space (left) and its embedding space after applying DA (right). The subscript “s” and “t” in the
legend refer to the source and target domains, respectively.

![Alt text](./figure/DAN.PNG)




Author
------------
Sungeun Hong / @[csehong][wp]
[wp]: sites.google.com/site/csehong


