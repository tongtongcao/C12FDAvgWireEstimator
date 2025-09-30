The package is for avergage-wire estimator for missing cluster on a track, developed by Transformer Masked Auto-Encoder (TMAE).
The package is designed by pytorch and pytorch-lightning.
Site-packages need to be installed:
  pip install torch torchvision
  pip install lightning
  pip install matplotlib
  pip install 
  pip install scipy
To run the package, python3 train.py

For the estimator, inputs are two paramters:
  1. Sequence of average wires for other 5 clusters
  2. Index of missing cluster ( = superlayer - 1)
output is average wire of missing cluster.

To apply the estimator in coatjava, an example for application of the estimator by ai.djl is developed.     
