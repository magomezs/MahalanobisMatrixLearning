# MahalanobisMatrixLearning

This repository include three caffe python layers, which are used to learnt a Mahalanobis Matrix in order to compare features with the Mahalanobis distance for person re-identification.

1. data_analysis_queues.py is a python module containing the caffe layer called DataAnalysisQueues.
2. data_analysis_weighted_average.py is a python module containing the caffe layer called DataAnalysisWeightedAverage.

These two layers estimate a Mahalanobis Matrix from the features corresponging to pairs of people images.

<br />
3. triplet_data_analysis_queues.py is a python module containing the caffe layer called DataAnalysisQueues, to estimate a Mahalanobis Matrix from the features corresponding to triplets of people images.

<br />

4. connection_function.py is a python module containing the caffe layer called ConnectionFunction.<br />
   This layer can computes two different distances, Euclidean and Mahalanobis distances, to compare the features corresponding    to pair of people images. 
<br />

# Citations

The layers 1 and 3 were described in the following paper as part of a Deep Parts Similarity Learning framework, DPSL. Please cite this work in your publications if it helps your research:

Gómez-Silva, M. J., Armingol, J. M., & de la Escalera, A. (2018). Deep Parts Similarity Learning for Person Re-Identification.  In Proceedings of the 13th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications (VISIGRAPP 2018) - Volume 5: VISAPP, pages 419-428.

@inproceedings{gomez2017deep,<br />
title={Deep Parts Similarity Learning for Person Re-Identification.},<br />
author={G{'o}mez-Silva, Mar{'\i}a Jos{'e} and Armingol, Jos{'e} Mar{'\i}a and de la Escalera, Arturo}, <br />
booktitle={In Proceedings of the 13th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications (VISIGRAPP 2018)}, <br />
pages={419--428}, <br />
year={2018} <br />
}
