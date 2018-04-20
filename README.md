# MahalanobisMatrixLearning

This repository include three caffe python layers, which are used to learnt a Mahalanobis Matrix in order to compare features with the Mahalanobis distance for person re-identification.


1. data_analysis_queues.py is a python module containing the caffe layer called DataAnalysisQueues.
2. data_analysis_weighted_average.py is a python module containing the caffe layer called DataAnalysisWeightedAverage.

These two layers estimate a Mahalanobis Matrix from the features corresponging to pairs of people images.


3. connection_function.py is a python module containing the caffe layer called ConnectionFunction.
   
   This layer can computes two different distances, Euclidean and Mahalanobis distances, to compare the features corresponding    to pair of people images. 


