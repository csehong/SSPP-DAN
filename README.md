SSPP-DAN in Tensorflow
====

Tensorflow implementation of [SSPP-DAN: Deep Domain Adaptation Network for Face Recognition with Single Sample Per Person][arxiv]
[arxiv]: https://arxiv.org/abs/1702.04069

![Alt text](./figure/overallflow.PNG)


Prerequisites
-------------
* [Python 2.7] [python]
* [Tensorflow 0.12] [tf] 
* [OpenCV 2.4.9] [cv]
* [NumPy] [np]
* [SciPy] [sp]
[python]: https://www.python.org/downloads/
[tf]: https://www.tensorflow.org/versions/r0.12/
[cv]: http://opencv.org/releases.html
[np]: http://www.numpy.org/
[sp]: https://www.scipy.org/install.html

###Usage
-------------



First, download the dataset from [our repository] [gd]
[gd]: https://drive.google.com/uc?id=0ByHRRxErVc0NTjFERTF5c1l2VVU&export=download

![Alt text](./EK-LFH.PNG)







ASDFA

```python
$ python download.py mnist celebA
import tensorflow as tf
```


To train a model with downloaded dataset:

$ python main.py --dataset mnist --input_height=28 --output_height=28 --train
$ python main.py --dataset celebA --input_height=108 --train --crop
To test with an existing model:

$ python main.py --dataset mnist --input_height=28 --output_height=28
$ python main.py --dataset celebA --input_height=108 --crop
Or, you can use your own dataset (without central crop) by:

$ mkdir data/DATASET_NAME
... add images to data/DATASET_NAME ...
$ python main.py --dataset DATASET_NAME --train
$ python main.py --dataset DATASET_NAME
$ # example
$ python main.py --dataset=eyes --input_fname_pattern="*_cropped.png" --train


