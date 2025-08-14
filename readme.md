# GPE Codes

## Description

This repository provides code for training Geometry-Preserving Encoder/Decoder (GPE)-based latent generative modeling on popular datasets such as MNIST, CIFAR10, CelebA, and CelebA-HQ (256x256). The implementation is modular and flexible, enabling users to easily adapt the code to experiment with other datasets of their choice. The code is written in Python and uses the PyTorch package for neural network training.

## Contents

- `GPE_train_encoder.py`: This script is used for training the GPE encoder.
- `GPE_train_decoder.py`: After training the encoder using `GPE_train_encoder.py`, this script can be used to train the GPE decoder.
- `GPE_train_CFM.py`: After training both the encoder and the decoder, this script can be used to train the flow map in the latent space based on the conditional flow matching algorithm. This part can be easily replaced with other flow-based generative models such as diffusion models or normalizing flows.
- `/utils`: A folder containing accessory Python scripts used in the code.
- `/transportmodules`: A folder containing the modules used for each dataset. Different neural network architectures can be easily implemented here.

## How to use (example with MNIST)

1. Clone the repository:
    ```bash
    git clone https://github.com/wonjunee/GPE_codes.git
    ```
2. Navigate to the project directory:
    ```bash
    cd GPE_codes
    ```
3. Train the encoder by running the following command:
    ```bash
    python GPE_train_encoder.py --data=mnist
    ```
4. Train the decoder by running the following command:
    ```bash
    python GPE_train_decoder.py --data=mnist
    ```
5. Train the flow map using the conditional flow matching algorithm by running the following command:
    ```bash
    python GPE_train_CFM.py --data=mnist
    ```

## Requirements

- PyTorch
- Matplotlib
- [NumPy](https://numpy.org/)
- [TQDM](https://tqdm.github.io/)
- [TorchCFM](https://github.com/atong01/conditional-flow-matching): The UNet implementation from this repository was used.


# Copyright & License Notice
Geometry Preserving Encoder/Decoder (GPE) is copyrighted by Regents of the University of Minnesota and covered by US 63/842,522. Regents of the University of Minnesota will license the use of GPE solely for educational and research purposes by non-profit institutions and US government agencies only. For other proposed uses, contact umotc@umn.edu. The software may not be sold or redistributed without prior approval. One may make copies of the software for their use provided that the copies, are not sold or distributed, are used under the same terms and conditions. As unestablished research software, this code is provided on an "as is'' basis without warranty of any kind, either expressed or implied. The downloading, or executing any part of this software constitutes an implicit agreement to these terms. These terms and conditions are subject to change at any time without prior notice.
