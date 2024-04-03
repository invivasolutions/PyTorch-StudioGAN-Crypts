# About
This markdown document provides an overview of the configurations set in the `Crypt-StyleGAN2-SPD-ADA.yaml` file, which is designed for training a StyleGAN2 model with specific modifications for the "Restained Crypts v3 Labels" dataset.

These configurations are read by `config.py` and utilized by `main.py` to set up and execute the training process, tailoring the behavior of the StyleGAN2 model to the specifics of the "Restained Crypts v3 Labels" dataset and desired training objectives.

This document serves as a guide to understanding and adjusting the configurations for similar or extended training scenarios.

## DATA Settings

-   name: Dataset name indicates the specific dataset used for training.
-   img_size: The size of images used for training. In this case, 512x512 pixels.
-   num_classes: The number of distinct classes in the dataset. Here, 4 classes are specified.

## MODEL Configurations

-   backbone: Specifies the backbone architecture of the model, which is StyleGAN2.
-   g_cond_mtd & d_cond_mtd: Methods for conditioning the generator (cAdaIN) and discriminator (SPD), affecting how input data is processed and features are learned.
-   g_act_fn & d_act_fn: Activation functions set to "Auto," allowing the model to adaptively choose the most suitable function.
-   z_prior: Defines the prior distribution of latent variables, here using a Gaussian distribution.
-   z_dim & w_dim: Dimensions of the latent space and intermediate latent space, respectively.
-   apply_g_ema: Whether to use an Exponential Moving Average for the generator weights, improving model stability over iterations.

## LOSS Mechanisms

-   adv_loss: The adversarial loss function, set to "logistic" for training.
-   apply_r1_reg: Indicates the use of R1 regularization to stabilize training.
-   r1_lambda & r1_place: Controls the strength and application phase of R1 regularization.

## OPTIMIZATION Details

-   batch_size: The number of samples per batch during training.
-   g_lr & d_lr: Learning rates for the generator and discriminator.
-   beta1 & beta2: Parameters for the Adam optimizer, controlling the decay rates of moving averages.
-   total_steps: Total number of training steps.

## AUG (Augmentations)

-   apply_ada: Enables Adaptive Discriminator Augmentation to enhance training with limited data.
-   ada_aug_type: Type of augmentation to apply, here "bgc" for background color modifications.
-   ada_initial_augment_p & ada_target: Initial augmentation probability and target probability for ADA.

## STYLEGAN Specifics

-   g_reg_interval & d_reg_interval: Intervals for applying regularization to the generator and discriminator.
-   mapping_network: Number of layers in the mapping network, affecting the complexity of latent space transformations.
-   style_mixing_p: Probability for mixing styles during training, encouraging diversity.
-   g_ema_kimg: Number of images after which the EMA of generator weights is updated.
-   apply_pl_reg: Whether to apply path length regularization, further stabilizing training.
-   pl_weight: Strength of the path length regularization.
-   d_architecture: Specifies the discriminator architecture, here "resnet" for resilience and performance.
-   d_epilogue_mbstd_group_size: Group size for minibatch standard deviation layers, enhancing diversity.

