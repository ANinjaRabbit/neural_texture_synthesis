# neural_texture_synthesis
Run `main.py` for synthesis using Gram method.

Run `main-gc.py` for synthesis using Guided Correspondence method.

It is advised not to use BF16 when synthesizing images with GC method.

We default to using FP16 for calculating the patch similarity and enabling
`torch.compile` for the loss calculation, which will significantly accelerate
the synthesis pipeline. Currently we did not provide flags to remove it. Feel
free to remove if your device does not support these features.

If `torch.compile` fails to find some CUDA headers, in a conda environment you
may run `conda install cuda-toolkit`.
