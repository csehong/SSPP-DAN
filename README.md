SSPP-DAN in Tensorflow
====

Tensorflow implementation of [SSPP-DAN: Deep Domain Adaptation Network for Face Recognition with Single Sample Per Person]

![Alt text](./figure/overallflow.PNG)



Prerequisites
-------------
* Python 2.7
* Tensorflow 0.12.1
* OpenCV
* NumPy
* SciPy

###Usage
-------------
First, download dataset with:

$ python download.py mnist celebA
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
