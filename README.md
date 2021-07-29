# Beacon
Beacon is a machine learning framework built on top of PyTorch.

## utils
utils contains utilities such as save load PyTorch checkpoints, load latest checkpoints, time counters, etc.

## nets
nets contains a base class for neural network training that encapsulates automatic model loading/saving.

## Sample
Run the following code for a minimal MNIST example that uses utils.

Training:  
`python examples/MNIST.py --mode train --expt-name MNISTTest --input-dir <DIR_TO_DATA> --output-dir <OUTPUT_DIR>`


Testing:  
`python examples/MNIST.py --mode test --expt-name MNISTTest --input-dir <DIR_TO_DATA> --output-dir <OUTPUT_DIR>`

# Contact
Srinath Sridhar  
[srinaths@umich.edu][1]

[1]: [mailto:srinaths@umich.edu]
