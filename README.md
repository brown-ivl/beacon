# Beacon
Beacon is a machine learning framework built on top of PyTorch.

## Requirements

- [pyTorch 1.1+]: pytorch is not automatically installed when you install beacon due to hardware differences, etc. Please see latest installation instructions on [pytorch.org][1]


## Installation

After installing the above requirements, you can install beacon like so:

`pip install git+https://github.com/brown-ivl/beacon.git`

If reinstalling, make sure to uninstall beacon before running the command (required on some systems).

## Sample
Run the following code for a minimal MNIST example classification example using an MLP.

Training:  
`python examples/classification_mlp.py --mode train --expt-name MNISTTest --input-dir <DIR_TO_DATA> --output-dir <OUTPUT_DIR>`


Testing:  
`python examples/classification_mlp.py --mode infer --expt-name MNISTTest --input-dir <DIR_TO_DATA> --output-dir <OUTPUT_DIR>`

# Contact
Srinath Sridhar  
[srinath@brown.edu][2]

[1]: https://pytorch.org/
[2]: [mailto:srinath@brown.edu]
