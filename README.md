# MahalanobisMatrixLearning

This repository includes a caffe python layer called maha_triplet_loss, which is a loss layer. This layer contains the re-formulation of the triplet loss function, by taking the Mahalanobis distance as the connection function between the features to compare, the formulation of the back-propagation stage of the new loss function, and the learning of the elements of the Mahalanobis matrix as an extra set of neural parameters.

1. maha_triplet_loss 

<br />
This also includes a layer to select triplets in order to speed up the learning procedure.

2. select_triplet


<br />
Moreover, this includes three caffe python layers, which are used to estimate a Mahalanobis Matrix in order to compare features with the Mahalanobis distance for person re-identification.

3. data_analysis_queues.py is a python module containing the caffe layer called DataAnalysisQueues.
4. data_analysis_weighted_average.py is a python module containing the caffe layer called DataAnalysisWeightedAverage.

These two layers estimate a Mahalanobis Matrix from the features corresponging to pairs of people images.

<br />

5. triplet_data_analysis_queues.py is a python module containing the caffe layer called DataAnalysisQueues, to estimate a        Mahalanobis Matrix from the features corresponding to triplets of people images.

<br />
In addition:
6. connection_function.py is a python module containing the caffe layer called ConnectionFunction.<br />
   This layer can computes two different distances, Euclidean and Mahalanobis distances, to compare the features corresponding to pair of people images. 
<br />

# Citations

The layers 3 and 6 were described in the following paper as part of a Deep Parts Similarity Learning framework, DPSL. Please cite this work in your publications if it helps your research:

Gómez-Silva, M. J., Armingol, J. M., & de la Escalera, A. (2018). Deep Parts Similarity Learning for Person Re-Identification.  In Proceedings of the 13th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications (VISIGRAPP 2018) - Volume 5: VISAPP, pages 419-428.

@inproceedings{gomez2017deep,<br />
title={Deep Parts Similarity Learning for Person Re-Identification.},<br />
author={G{'o}mez-Silva, Mar{'\i}a Jos{'e} and Armingol, Jos{'e} Mar{'\i}a and de la Escalera, Arturo}, <br />
booktitle={In Proceedings of the 13th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications (VISIGRAPP 2018)}, <br />
pages={419--428}, <br />
year={2018} <br />
}

Layers 3 and 4 where compared in the following paper. Please cite this work in your publications if it helps your research:

M. J. Gómez Silva, J. M. Armingol, and A. de la Escalera, “Re-identificación de personas mediante la distancia de mahalanobis,” Actas de las XXXIX Jornadas de Automática, Badajoz, 5-7 de Septiembre de 2018, 2018.

@article{gomez2018re,<br />
  title={Re-identificaci{\'o}n de personas mediante la distancia de Mahalanobis},<br />
  author={G{\'o}mez Silva, Mar{\'\i}a J and Armingol, Jos{\'e} Mar{\'\i}a and de la Escalera, Arturo},<br />
  journal={Actas de las XXXIX Jornadas de Autom{\'a}tica, Badajoz, 5-7 de Septiembre de 2018},<br />
  year={2018},<br />
  publisher={Universidad de Extremadura}<br />
}


Layers 3 and 5 where compared in the following paper. Please cite this work in your publications if it helps your research:

M. J. Gómez-Silva, J. M. Armingol, and A. de la Escalera, “Balancing people re-identification data for deep parts similarity learning,” Journal of Imaging Science and Technology, 2019.

@article{gomez2019balancing,<br />
  title={Balancing people re-identification data for deep parts similarity learning},<br />
  author={G{\'o}mez-Silva, Mar{\'\i}a Jos{\'e} and Armingol, Jos{\'e} Mar{\'\i}a and Escalera, Arturo de la},<br />
  journal={Journal of Imaging Science and Technology},<br />
  volume={63},<br />
  number={2},<br />
  pages={20401--1},<br />
  year={2019},<br />
  publisher={Society for Imaging Science and Technology}<br />
}<br />


# ACKNOWLEDGMENTS
This work was supported by the Spanish Government through the CICYT projects (TRA2015-63708-R and TRA2016-78886-C3-1-R), and Ministerio de Educación, Cultura y Deporte para la Formación de Profesorado Universitario (FPU14/02143), and Comunidad de Madrid through SEGVAUTO-TRIES (S2013/MIT- 2713). We gratefully acknowledge the support of NVIDIA Corporation with the donation of the GPUs used for this research.
