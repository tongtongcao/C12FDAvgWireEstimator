The package is to train avergage-wire estimator for missing cluster on a track by Transformer Masked Auto-Encoder (TMAE).\
The AI model is designed by pytorch and pytorch-lightning.\
Before run the package, site-packages need to be installed if CPU only:
- pip install torch torchvision
- pip install lightning
- pip install matplotlib
- pip install scipy

If run the package with GPU, cuda needs to be installed.
  
To run the package,
- python3 train.py ...

Arguments:
  - positional arguments:
    - inputs      &nbsp;&nbsp;&nbsp;          One or more input CSV files (default: avgWires.csv)

  - options:
    - -h, --help     &nbsp;&nbsp;&nbsp;       show this help message and exit
    - --device {cpu,gpu,auto} &nbsp;&nbsp;&nbsp; Choose device: cpu, gpu, or auto (default: auto)
    - --max_epochs MAX_EPOCHS &nbsp;&nbsp;&nbsp; Number of training epochs
    - --batch_size BATCH_SIZE &nbsp;&nbsp;&nbsp; Batch size for DataLoader
    - --outdir OUTDIR   &nbsp;&nbsp;&nbsp;    Directory to save models and plots
    - --end_name END_NAME &nbsp;&nbsp;&nbsp;  Optional suffix to append to output files (default: none)
    - --d_model D_MODEL  &nbsp;&nbsp;&nbsp;   Transformer embedding dimension (must be divisible by nhead)
    - --nhead NHEAD   &nbsp;&nbsp;&nbsp;      Number of attention heads in the transformer (default: 4)
    - --num_layers NUM_LAYERS &nbsp;&nbsp;&nbsp; Number of transformer encoder layers (default: 2)
    - --lr LR        &nbsp;&nbsp;&nbsp;       Learning rate for optimizer (default: 1e-3)
    - --no_train      &nbsp;&nbsp;&nbsp;      Skip training and only run inference using a saved model
    - --enable_progress_bar &nbsp;&nbsp;&nbsp; Enable progress bar during training (default: disabled)

For the estimator, inputs are two paramters:
  1. Sequence of average wires for other 5 clusters
  2. Index of missing cluster ( = superlayer - 1)
     
and output is average wire of missing cluster.

To apply the estimator in coatjava, an example for application of the estimator by ai.djl is developed.
- Install: mvn clean install
- Run: java -cp "target/TestDJL-1.0-SNAPSHOT.jar:target/lib/*" org.example.Main
