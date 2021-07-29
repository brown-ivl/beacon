# Beacon
Beacon is a machine learning framework built on top of PyTorch.

## Requirements

- [pyTorch 1.1+]: pytorch is not automatically installed when you install beacon due to hardware differences, etc. Please see latest installation instructions on [pytorch.org][1]


## Installation

After the above requirements are installed, you can install tk3dv like so:

`pip install git+https://github.com/brown-ivl/beacon.git`

If reinstalling, make sure to uninstall and repeat the install (required on some systems).

## Sample
Run the following code for a minimal MNIST example that uses utils.

Training:  
`python examples/MNIST.py --mode train --expt-name MNISTTest --input-dir <DIR_TO_DATA> --output-dir <OUTPUT_DIR>`


Testing:  
`python examples/MNIST.py --mode test --expt-name MNISTTest --input-dir <DIR_TO_DATA> --output-dir <OUTPUT_DIR>`

# Contact
Srinath Sridhar  
[srinaths@umich.edu][2]

[1]: https://pytorch.org/
[2]: [mailto:srinaths@umich.edu]
