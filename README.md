The package is to train avergage-wire estimator for missing cluster on a track by Transformer Masked Auto-Encoder (TMAE).\
The AI model is designed by pytorch and pytorch-lightning.\
Before run the package, site-packages need to be installed:
- pip install torch torchvision
- pip install lightning
- pip install matplotlib
- pip install scipy
  
To run the package,
- python3 train.py

For the estimator, inputs are two paramters:
  1. Sequence of average wires for other 5 clusters
  2. Index of missing cluster ( = superlayer - 1)
     
and output is average wire of missing cluster.

To apply the estimator in coatjava, an example for application of the estimator by ai.djl is developed.
- Install: mvn clean install
- Run: java -cp "target/TestDJL-1.0-SNAPSHOT.jar:target/lib/*" org.example.Main
