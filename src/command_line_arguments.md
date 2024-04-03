# Additional Information

- The configuration file examples are located in the "configs" folder of the repository. Any parameters defined in the configuration files override the default values and any arguments set via the command line. This allows for easy adjustments and experimentation by modifying the configuration files without the need to change the command line arguments for each run.

# Configuration Options

This document outlines the configuration options available for the package. These options are set via command-line arguments when running the application.

## Logging Options

- `wandb` logging (`--entity`, `--project`): Weights & Biases (wandb) is a tool for tracking experiments, visualizing data, and sharing insights about machine learning projects. Specify the `entity` as your username or team name in wandb, and `project` as the name of the project you're working on.

## Path Options

- The `-cfg`, `--cfg_file` specifies the path to a YAML or JSON configuration file, allowing users to manage configurations separately from the command line interface. This is useful for setting up experiments with different parameters.
- `-data`, `--data_dir` is the directory where your dataset is located. This path is crucial for loading your data correctly.
- `-save`, `--save_dir` determines where to save output files, such as models and logs, during training or evaluation.
- `-ckpt`, `--ckpt_dir` and `-best`, `--load_best` are related to model checkpoints. The former specifies where to save or load model checkpoints, and the latter indicates whether to load the best-performing model checkpoint based on some validation metric.

## Training Options

- Setting a `--seed` ensures reproducibility in experiments by initializing the random number generators of Python, NumPy, and PyTorch with the same seed.
- `-DDP`, `--distributed_data_parallel` enables Distributed Data Parallel training, which is a method of parallelizing data across multiple GPUs, potentially across several nodes, to accelerate training.
- `--backend` specifies the backend used for inter-GPU communication during distributed training, with "nccl" being optimized for NVIDIA GPUs and "gloo" being a more cross-platform option.

## GAN-Specific Options

- The `--truncation_factor` and `--truncation_cutoff` are techniques used to regulate the diversity-quality tradeoff in generated images by modulating the latent space.
- `-batch_stat`, `--batch_statistics` and `-std_stat`, `--standing_statistics` are strategies for evaluating Generative Adversarial Networks (GANs) more effectively by either using batch statistics or "standing" statistics over multiple batches for normalization layers.
- `-std_max`, `--standing_max_batch`, and `-std_step`, `--standing_step` further refine the standing statistics calculation by limiting the batch size and the number of steps over which statistics are accumulated.
- `--freezeD` is used in transfer learning scenarios, where parts of the discriminator are frozen to transfer knowledge from one task to another without the need for retraining those parts.

## Advanced Features

- **Synchronized Batch Normalization**: When using multiple GPUs, it's crucial to synchronize the mean and variance computation of batch normalization layers across all devices. This ensures that each device normalizes its inputs using the global mean and variance, which is especially important for consistency in batch statistics when training models in a distributed manner.
- **Mixed Precision Training** (`-mpc`, `--mixed_precision`): Utilizes floating-point 16 (FP16) and 32 (FP32) to speed up training, reduce memory usage, and maintain the model's accuracy. It's particularly effective on GPUs with Tensor Cores that accelerate FP16 computations.


## Langevin Sampling Options

- `-lgv`, `--langevin_sampling`: Apply langevin sampling to generate images from an Energy-Based Model (EBM). Langevin sampling helps in drawing samples from complex distributions and is particularly useful in the context of GANs for generating high-quality images.
- `-lgv_rate`, `--langevin_rate`: Initial update rate for langevin sampling (\(\epsilon\)). Type: `float`. Default: `-1` (means no update rate is applied).
- `-lgv_std`, `--langevin_noise_std`: Standard deviation of the Gaussian noise used in langevin sampling (std of \(n_i\)). This parameter controls the amount of noise added during each sampling step. Type: `float`. Default: `-1`.
- `-lgv_decay`, `--langevin_decay`: Decay strength for `langevin_rate` and `langevin_noise_std`, allowing these parameters to decrease over time to fine-tune the sampling process. Type: `float`. Default: `-1` (no decay is applied).
- `-lgv_decay_steps`, `--langevin_decay_steps`: Specifies how often (in steps) the `langevin_rate` and `langevin_noise_std` decrease. Type: `int`. Default: `-1`.
- `-lgv_steps`, `--langevin_steps`: Total steps of langevin sampling to perform. Type: `int`. Default: `-1` (indicating a default or unspecified number of steps).

## Training and Data Loading Options

- `-t`, `--train`: Flag to indicate that the model should be trained.
- `-hdf5`, `--load_train_hdf5`: Load train images from an HDF5 file for fast I/O. This is particularly useful when working with large datasets.
- `-l`, `--load_data_in_memory`: Load the entire training dataset into main memory to speed up training by reducing disk I/O.

## Evaluation and Image Processing Options

- `-metrics`, `--eval_metrics`: Evaluation metrics to use during training. Possible values include `fid` (Fréchet Inception Distance), `is` (Inception Score), `prdc` (Precision, Recall, Density, and Coverage). Allows for a list of metrics. Default: `['fid']`.
- `--pre_resizer` and `--post_resizer`: Specify the image resizing algorithm to be used before and after processing, respectively. Options include 'wo_resize' (without resize), 'nearest', 'bilinear', 'bicubic', 'lanczos' for `pre_resizer`, and 'legacy', 'clean', 'friendly' for `post_resizer`.
- `--num_eval`: Number of runs for final evaluation to ensure stability and reliability of the results. Type: `int`. Default: `1`.

## Image Saving and Visualization Options

- `-sr`, `--save_real_images` and `-sf`, `--save_fake_images`: Flags to save images sampled from the reference dataset and fake images generated by the GAN, respectively.
- `-sf_num`, `--save_fake_images_num`: Number of fake images to save. This is useful for generating a gallery or for detailed analysis of the generated images. Type: `int`. Default: `1`.
- `-v`, `--vis_fake_images`: Visualize image canvas to inspect the quality and diversity of generated images.
- `-knn`, `--k_nearest_neighbor`: Perform k-nearest neighbor analysis to evaluate the diversity and coverage of the generated images compared to the real dataset.
- `-itp`, `--interpolation`: Conduct interpolation analysis to understand the smoothness and transitions between different points in the latent space.

## Analysis Options

- `-fa`, `--frequency_analysis`: Conduct frequency analysis to examine the distribution of frequencies in the generated images. This can help identify issues like mode collapse or check the diversity of the generated images.
- `-tsne`, `--tsne_analysis`: Perform t-SNE analysis to visualize the high-dimensional data of generated images in a two-dimensional space. It's useful for understanding the clustering and diversity of the images.
- `-ifid`, `--intra_class_fid`: Calculate intra-class FID (Fréchet Inception Distance) to evaluate the quality of generated images within specific classes. It's a variation of the FID metric that focuses on the diversity and quality within each class.
- `--GAN_train` and `--GAN_test`: Flags for calculating CAS (Class Activation Sequences) for Recall and Precision, respectively. These are metrics for evaluating the diversity and fidelity of generated images in a class-aware manner.
- `-resume_ct`, `--resume_classifier_train`: Whether to resume training of the classifier for CAS. This is relevant for experiments where a classifier is trained as part of the GAN evaluation process.
- `-sefa`, `--semantic_factorization`: Perform semantic (closed-form) factorization to discover and manipulate latent semantic factors in a learned model. SEFA allows for interpretable control over generated content.
- `-sefa_axis`, `--num_semantic_axis`: Number of semantic axes to explore via SEFA. This determines the number of latent directions identified for manipulation. Type: `int`. Default: `-1`.
- `-sefa_max`, `--maximum_variations`: Specifies the range for interpolating between \(z\) and \(z + 	ext{maximum_variations} 	imes 	ext{eigen-vector}\). This controls how much variation is introduced along each semantic axis identified by SEFA. Type: `float`. Default: `-1`.

## Performance and Utility Options

- `-empty_cache`, `--empty_cache`: Empty CUDA caches after each training step of the generator and discriminator. While it may slightly reduce memory usage, it can slow down training speed and is not recommended for normal use.
- `--print_freq`: Logging interval, determining how often to print logs during training. Type: `int`. Default: `100`.
- `--save_freq`: Save interval, specifying how often to save checkpoints of the model during training. Type: `int`. Default: `2000`.
- `--eval_backbone`: Specifies the backbone model to use for evaluation metrics like FID and Inception Score. Options include various versions of InceptionV3 and ResNet50, among others. Default: `InceptionV3_tf`.
- `-ref`, `--ref_dataset`: Reference dataset to use for evaluation, can be set to `train`, `valid`, or `test`. This determines which dataset partition is used as the ground truth for evaluating generated images.
- `--calc_is_ref_dataset`: Whether to calculate the Inception Score of the reference dataset. This can provide a baseline measure of the diversity and quality of the real images against which generated images are compared.