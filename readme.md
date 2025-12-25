# Geometry-Preserving Encoder/Decoder in Latent Generative Models

**Authors:**  
- Wonjun Lee\*  
- Riley C. W. O'Neill\*  
- Dongmian Zou\*\*  
- Jeff Calder\*  
- Gilad Lerman\*

\* University of Minnesota  
\*\* Duke Kunshan University

## Description

This repository provides code for training Geometry-Preserving Encoder/Decoder (GPE)-based latent generative modeling on popular datasets such as MNIST, CIFAR10, CelebA, and CelebA-HQ (256x256). The implementation is modular and flexible, enabling users to easily adapt the code to experiment with other datasets of their choice. The code is written in Python and uses the PyTorch package for neural network training.

Link to the paper: [https://arxiv.org/abs/2501.09876](https://arxiv.org/abs/2501.09876)



## Requirements

- PyTorch
- Matplotlib
- [NumPy](https://numpy.org/)
- [TQDM](https://tqdm.github.io/)
- [TorchCFM](https://github.com/atong01/conditional-flow-matching): The UNet implementation from this repository was used.


## Contents

The repository is organized around a modular training pipeline for geometry-preserving encoders and latent-space generative modeling.

- **`GPE_train_encoder.py`**  
  This script is used to train the **GPE encoder**, which maps data from the ambient space into a latent space while enforcing geometry-preserving properties.  
  The encoder is optimized using the GPE objective, which balances reconstruction fidelity with invariance and geometric consistency under data perturbations or corruptions.  
  The trained encoder defines the latent representation used in all subsequent stages.

- **`GPE_train_decoder.py`**  
  After the encoder has been trained using `GPE_train_encoder.py`, this script is used to train the **GPE decoder**.  
  The decoder maps latent variables back to the ambient data space, ensuring consistency with the fixed encoder and enabling reconstruction or generation.  
  Training the decoder separately preserves the geometric properties learned by the encoder and simplifies both implementation and analysis.

- **`GPE_train_CFM.py`**  
  After training both the encoder and decoder, this script is used to train a **latent-space generative model** based on the conditional flow matching (CFM) algorithm.  
  The generative model operates entirely in the latent space, which typically improves conditioning and accelerates training.  
  This component is modular and can be easily replaced by other flow-based generative models such as diffusion models or normalizing flows.

- **`utilfunctions.py`**
This file contains **shared utility functions** that support all stages of the training pipeline, including encoder training, decoder training, and latent-space generative modeling.  
The functions in this file are intentionally kept model-agnostic and provide common numerical, statistical, and training-related operations used throughout the codebase.


- **`/utils/`**  
  A directory containing **utility and accessory scripts** shared across the codebase.  

- **`/transportmodules/`**  
  A directory containing **dataset-specific transport modules**, including neural network architectures and data-dependent components.  
  Each module defines the encoder, decoder, and latent generative model appropriate for a given dataset.  
  New datasets or architectures can be added by implementing additional modules in this directory, without modifying the main training scripts.


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

# Copyright & License Notice
**Geometry Preserving Encoder/Decoder (GPE)** is copyrighted by **Regents of the University of Minnesota** and covered by US 63/842,522. Regents of the University of Minnesota will license the use of GPE solely for educational and research purposes by non-profit institutions and US government agencies only. For other proposed uses, contact umotc@umn.edu. The software may not be sold or redistributed without prior approval. One may make copies of the software for their use provided that the copies, are not sold or distributed, are used under the same terms and conditions. As unestablished research software, this code is provided on an "as is'' basis without warranty of any kind, either expressed or implied. The downloading, or executing any part of this software constitutes an implicit agreement to these terms. These terms and conditions are subject to change at any time without prior notice.
