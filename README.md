# Melody extraction using joint detection and classification network
The source code of *"Joint Detection and Classification of Singing Voice Melody Using Convolutional Recurrent Neural Networks"*, Applied Sciences (2019) |<a href = "https://www.mdpi.com/2076-3417/9/7/1324" target="_blank">PDF</a>|


## Abstract

We present a joint detection and classification (JDC) network that conducts the singing voice detection and the pitch estimation simultaneously. The JDC network is composed of the main network that predicts the pitch contours of the singing melody and anauxiliary network that facilitates the detection of the singing voice. 

<img src="./img/diagram.png" width="70%">

The main network is built with a convolutional recurrent neural network with residual connections and predicts pitch labels that cover the vocal range with a high resolution as well as non-voice status. 
The auxiliary network is trained to detect the singing voice using multi-level features shared from the main network. The two optimizations processed are tied with a joint melody loss function. 

We evaluate the proposed model on multiple melody extraction and vocal detection datasets, including cross-dataset evaluation. 
The experiments demonstrate how the auxiliary network and the joint melody loss function improve the melody extraction performance. Also, the results show that our method outperforms state-of-the-art algorithms on the datasets.


## Dependencies

- OS : LINUX 

- Programming language : Python 3.6+

- Python Library 
  - Keras 2.2.2 (Deep Learning library)
  - Librosa 0.6.2 (for STFT)  
  - madmom 0.16.1 (for loading audio and resampling)
  - Numpy, SciPy

-  Hardware
  -  2 GPU : GeForce GTX 1080ti
  
