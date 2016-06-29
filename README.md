# DeepList
This is the source code of our TCSVT paper:
<DeepList: Learning Deep Features with Adaptive Listwise Constraint for Person Re-identification>


The package contains the following components:

(1)/code

The source code for the proposed DeepList method. We use the deep learning toolbox MatConvNet and add several layers.

(2)/demo_viper

The demo code for VIPeR dataset. We provide the finetuning code based on a pretrained model on Market1501 and CUHK03 dataset.

(3)/mat

Some mat files needed to run the code. 

To evaluate the provided code, please first download the datafile and pretrained model from http://pan.baidu.com/s/1dE8LzLv, and put them in the folder /mat, then run /demo_viper/VIPeR_full_BN_fineTune.m to train the model for full-scale image. The final reported results are obtained by ensemble four models trained on full-scale, half-scale, top part and middle part images. Please refer to our paper for more details.

If you have any questions, please contact me: jinw@hust.edu.cn
