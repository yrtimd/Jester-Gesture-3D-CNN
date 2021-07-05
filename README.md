
# Jester Gesture Dataset Training & Testing Environment

### Source Github 
The original github, which this work is based on, can be found [here](https://github.com/udacity/CVND---Gesture-Recognition). 

## About

The Jester Gesture Dataset Training & Testing Environment is a python notebook that aims to simplify the source github repository into a more readable and explainable environment. The Notebook includes the ability to train and test models on the jester v1 dataset. Additionally, playable video examples (including live predictions) and the ability to process and predict camera input are provided in the notebook.


## Jester Dataset
The [Jester v1 Dataset](https://20bn.com/datasets/jester) by TwentyBN contians 27 pre-defined gesture composed of 148,092 videos, of which 14,743 are unlabeled for testing purposes. The videos vary in terms of background, lighting and quality making the data non-uniform.

The dataset can be downloaded with the link above; a license is required to be signed but it is mostly for formality (for individuals and academia). The download size is 22.8GB of compressed data and ~54GB of uncompressed data. Additionally, the download consists of 23 1 GB files, which have to be unzipped together with the following command <code>cat 20bn-jester-v1-?? | tar zx</code>. 

## Notebook Usage

The notebook only needs to be launched and has additional explanations and instruction inside.
The notebook is split into 5 main parts:
- Data Loading
- Model, Training & Testing Definitions
- Training + Plotting
- Testing
- Video Examples

### Model

The notebook comes with a basic 3D CNN models (7 layers total) and 20 epoch pre-trained model that can be selected. The model can be changed in the Model cell in the Python Notebook, however, chaning the input of the model requires the changing of multiple parameters and thus is not recommended.

## References

`Zhu, G., Zhang, L., Mei, L., Shen, P., Shah, S.A.A. and Bennamoun, M.  (2018) _Attention in Convolutional LSTM for Gesture Recognition._ In: 32nd Conference on Neural Information Processing Systems (NIPS) 2018, 3 - 8 December 2018, Montreal, Canada`

`J. Materzynska, G. Berger, I. Bax and R. Memisevic, "The Jester Dataset: A Large-Scale Video Dataset of Human Gestures," 2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW)`

`L. Shi, Y. Zhang, J. Hu, J. Cheng and H. Lu, "Gesture Recognition Using Spatiotemporal Deformable Convolutional Representation," _2019 IEEE International Conference on Image Processing (ICIP)`

