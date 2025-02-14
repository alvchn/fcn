This repository contains Python code for recurrent fully convolutional network (RFCN) models from our paper "Deep learning robotic guidance for autonomous vascular access" (https://www.nature.com/articles/s42256-020-0148-7).

## Dependencies

The models were developed using Python3 with Tensorflow 1.4. The Python dependencies for running the code include

```
Tensorflow 1.4
SimpleITK
Matplotlib
ImageIO
Numpy
```

## Data preparation

Sample test data are provided in the folders ```/data/nir_test``` and ```/data/dus_test```. Each directory contains multiple sets of test images. Individual data are in the form of .mat files, structured as follows:

```
/data/nir_test/sequenceX/data_YYYYYY.mat
  Image_left - WxH rectified left NIR stereo image input.
  Image_right - WxH rectified right NIR stereo image input.
  Mask_left - WxH optional binary input mask of the left arm segmentation.
  Mask_right - WxH optional binary input mask of the right arm segmentation.
  Label_left - WxH left binary segmentation label.
  Label_right - WxH right binary segmentation label.
  Disparity - WxHx1x2 stereo disparity map label.
```

Output predictions may be (optionally) written to disk as .png image files.

## Train and test models from scratch

Python scripts for training models from scratch are included under ```/nir_trainer``` and ```/dus_trainer```. Both follow a similar set up:

```nir_setup.py``` and ```dus_setup.py``` parse the data into training, validation, and test splits. The script use a list structure given by

```
datalist = [[dataPath1, startingFrameIndex, endingFrameIndex, timeSteps], 
            [dataPath2, startingFrameIndex, endingFrameIndex, timeSteps],
            ...
           ]
```

For single frame model (FCN), data txt files should be a list of image file names separated by line breaks. For recurrent model (Rec-FCN), data txt files should be a list of image file names separated by line breaks, which each image sequence is separated by a dash ("-"), i.e.,

```
relative_path\sequence1\data_000001.mat
relative_path\sequence1\data_000002.mat
relative_path\sequence1\data_000003.mat
-
relative_path\sequence2\data_000001.mat
relative_path\sequence2\data_000002.mat
relative_path\sequence2\data_000003.mat
-
relative_path\sequence3\data_000001.mat
relative_path\sequence3\data_000002.mat
relative_path\sequence3\data_000003.mat
```

Separate data lists can be created to handle training, validation, and testing. Example data lists can be found under ```/nir_trainer/datalists``` and ```/dus_trainer/datalists```.

```nir_run.py``` and ```dus_run.py``` are the entry points for running the models. These scripts set the relevant paths needed for model training and testing and allow the user to set input parameters that define the model structure and training strategy. For example,

```nir_trainer.py``` and ```dus_trainer.py``` are the main classes that set up the model structure according to the user inputs, build the Tensorflow model graph, create the batches, perform online data augmentation, and run the training/test sessions. 

The main function definitions within these classes for model training and testing are provided below (along with default input values):

```
class ModelTrainer:
    def train(self, training_list,
                    validation_list,
                    testing_list,
                    model_type = 'RFCN',
                    model_checkpoint = None, 
                    restartdirectoryname = None,
                    restart_from = -1,
                    epochs = 101,
                    epochs_save = 5,
                    learning_rate = 0.00001,
                    loss_type='WCE+Dice+MSE',
                    loss_weights = (1.0, 0.0, 0.0, 0.0, 0.0),
                    pos_weight = 3.0,
                    threshold = 0.5,
                    k_composite=(0.5, 0.5, 0.5, 0.5, 0.5),
                    min_max=(0, 1, 0, 1, 0, 1, 0, 1, 0, 1),
                    n_augs = 1,
                    batch_size = 8,
                    batch_size_val = 8,
                    window_size = 1,
                    input_channels = 2,
                    output_channels = 2,
                    n_channels = 4,
                    crop_from_seg_center = False,
                    apply_augmentation = False,
                    shuffle_sequence_order = True,
                    apply_reverse_order = True,
                    predict_from_label = False,
                    write_prediction_text_file = False,
                    write_images = False,
                    show_images = False,
                    show_activation_maps = False
                    ):
                    
    def test(self, testing_list,
                   model_type = 'RFCN',
                   model_checkpoint = None, 
                   restartdirectoryname = None,
                   restart_from = -1,
                   loss_type = 'WCE',
                   loss_weights = (1.0, 0.0, 0.0, 0.0, 0.0),
                   pos_weight = 3.0,
                   threshold = 0.5,
                   k_composite = (0.5, 0.5, 0.5, 0.5, 0.5),
                   min_max = (0, 1, 0, 1, 0, 1, 0, 1, 0, 1),
                   n_augs = 1,
                   batch_size = 8,
                   window_size = 1,
                   input_channels = 2,
                   output_channels = 2,
                   n_channels = 4,
                   crop_from_seg_center = False,
                   apply_augmentation = False,
                   predict_from_label = False,
                   write_prediction_text_file = False,
                   write_images = False,
                   show_images = False,
                   show_activation_maps = False
                   ):
```

Other class functions for training and testing batches:

```
    def createMiniBatch() # creates mini-batches for training
    def createMiniBatchForTesting() # creates mini-batches for testing
    def computeCompositeTransforms() # creates the composite transforms for data augmentation using SimpleITK
    def getResults() # handles data writing, data display, and loss calculation during training and testing
```

The model definitions are found in fcn_model/fcn_model.py and includes the following network layer definitions:

```
def prelu() # parametric leaky Relu nonlinear activation function

def convolution_2d() # standard convolution operator
def deconvolution_2d() # standard transpose convolution operator
def convolution_block() # convolution block (3x3 kernel size) (used in encoder)
def convolution_block_2() # convolution block (3x3 kernel size) preceded by skip connection (used in decoder)
def convolution_block_in() # convolution block (variable kernel size) (used in encoder)
def convolution_block_2_in() # convolution block (variable kernel size) preceded by skip connection (used in decoder)
def convolution_block_in_batchnorm() # convolution block (variable kernel size) with batch norm layers (used in encoder)
def convolution_block_2_in_batchnorm() # convolution block (variable kernel size) with batch norm layers and preceded by skip connection (used in decoder)

def down_convolution() # stride-2 convolutions for down sampling (used in encoder)
def up_convolution() # stride-2 tranpose convolutions for up sampling (used in decoder)
def up_convolution_resize() # stride-2 resize-up convolutions for up sampling (used in decoder)

def fcn_encoder() # encoder
def fcn_encoder_batchnorm() # encoder with batch normalization
def fcn_decoder() # decoder
def fcn_decoder_batchnorm() # decoder with batch normalization
def fcn_encoder_decoder() # encoder-decoder

def convGRU_2d_gate() # convolutional gated recurrent unit
def convGRU_2d_output() # convolutional gated recurrent unit output
def fcn_convGRU() # convolutional GRU structure

def fcn_outer() # overall encoder-decoder structure
def fcn_outer_batchnorm() # overall encoder-decoder structure with batch normalization
def rfcn_outer() # overall recurrent encoder-decoder structure
def rfcn_outer_batchnorm() # overall recurrent encoder-decoder structure with batch normalization
```

Loss functions used in the reported studies are defined in /fcn_model/fcn_lossfunc.py:

```
def sigmoid()
def computeModifiedHausdorffDistance2D_TF()
def computeModifiedHausdorffDistance2D()
def computeWeightedCrossEntropyWithLogits_TF()
def computeWeightedCrossEntropyWithLogits()
def computeDiceLoss_TF()
def computeGeneralizedDiceLoss2D_TF()
def computeGeneralizedDiceLoss_TF()
def computeDiceScorePrediction()
def computeMeanSquaredLoss_TF()
def computeMeanSquaredLoss()
def setLoss_TF()
def evalLoss_TF()
def evalLoss_customized()
```

Finally, a number of utility functions are provided in ```/fcn_model/fcn_utilities.py```:

```
def showActivationMaps() # show network activation maps at each spatial resolution layer
def showImagesAsSubplots() # show input and output images as subplots
def writeOutputsToFile() # write network predictions to specified file path
```

## References

Tensorflow repositories:

Tensorflow VNet
```
@misc{jackyko1991_vnet_tensorflow,
  author = {Jacky KL Ko},
  title = {Implementation of vnet in tensorflow for medical image segmentation},
  howpublished = {\url{https://github.com/jackyko1991/vnet-tensorflow}},
  year = {2018},
  publisher={Github},
  journal={GitHub repository},
}
```

Tensorflow UNet
```
@article{akeret2017radio,
  title={Radio frequency interference mitigation using deep convolutional neural networks},
  author={Akeret, Joel and Chang, Chihway and Lucchi, Aurelien and Refregier, Alexandre},
  journal={Astronomy and Computing},
  volume={18},
  pages={35--39},
  year={2017},
  publisher={Elsevier}
}
```

Tensorflow convolutional LSTM/GRU cell
```
@article{carlthome,
  author = {carlthome},
  title = {A ConvLSTM cell for TensorFlow's RNN API},
  howpublished = {\url{https://github.com/carlthome/tensorflow-convlstm-cell}},
  year={2017},
  publisher={Github},
  journal={GitHub repository},
}
```

Good references for CNN/ConvRNN models
```
@article{meiburger2017,
  author = {Meiburger, Kristen M and Acharya, U Rajendra and Molinari, Filippo},
  title = {Automated localization and segmentation techniques for B-mode ultrasound images: a review},
  howpublished = {\url{https://www.sciencedirect.com/science/article/abs/pii/S0010482517303888}},
  year={2018},
}
```

```
@article{chen2016,
  author = {Chen, Jianxu and Yang, Lin and Zhang, Yizhe and Alber, Mark and Chen, Danny Z},
  title = {Combining fully convolutional and recurrent neural networks for 3D biomedical image segmentation},
  howpublished = {\url{https://arxiv.org/abs/1609.01006}},
  year={2016},
}
```

```
@article{gao2017,
  author = {Gao, Yang and Phillips, Jeff M and Zheng, Yan and Min, Renqiang and Fletcher, Thomas P and Gerig, Guido},
  title = {Fully convolutional structured LSTM networks},
  howpublished = {\url{https://arxiv.org/pdf/1609.01006.pdf}},
  journal={2018 IEEE 15th International Symposium on Biomedical Imaging (ISBI 2018)},
  year={2017},
  publisher={IEEE}
}
```

```
@article{milletari2018,
  author = {Milletari, Fausto and Rieke, Nicola and Baust, Maximilian and Esposito, Marco and Navab, Nassir},
  title = {CFCM: segmentation via coarse to fine context memory},
  howpublished = {\url{https://arxiv.org/abs/1806.01413}},
  year={2018},
}
```

```
@article{arbelle2019,
  author = {Arbelle, Assaf and Raviv, Tammy Riklin},
  title = {Microscopy cell segmentation via convolutional LSTM networks},
  howpublished = {\url{https://arxiv.org/abs/1805.11247}},
  year={2019},
}
```

```
@article{ni2020,
  author = {Hao Ni},
  title = {Deep learning for 4D longitudinal segmentation of MRI brain tissues and glioma},
  howpublished = {\url{https://repository.tudelft.nl/islandora/object/uuid%3Ae34a8dee-0bdb-4e79-9d42-3fc3998bbb23}},
  year={2020},
}
```

Papers that made use of parts of this code:
```
@article{chen2020,
  author = {Chen, Alvin and Balter, Max and Maguire, Tim and Yarmush, Martin L},
  title = {Deep learning robotic guidance for autonomous vascular access},
  howpublished = {\url{https://doi.org/10.1038/s42256-020-0148-7}},
  year={2020},
  publisher={Springer}
}
```

```
@article{lee2020,
  author = {Lee, Brian C and Vaidya, Kunal and Jain, Ameet K and Chen, Alvin},
  title = {Guidewire segmentation in 4D ultrasound sequences using recurrent fully convolutional networks},
  howpublished = {\url{https://link.springer.com/chapter/10.1007/978-3-030-60334-2_6}},
  year={2020},
  publisher={Springer}
}
```
