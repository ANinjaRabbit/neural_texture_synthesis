# Neural Texture Synthesis
## Environment
The project depends on PyTorch, OpenCV and TorchVision. `tqdm`, `scipy`, and
`matplotlib` may also be required. `torch.compile` may require CUDA toolkit.

We suggest preparing the environment using Conda. Run
```bash
conda env create -f environment.yml -n ENV_NAME
conda activate ENV_NAME
```

## Synthesis workflow
Run `main.py` or `main-gc.py` for texture synthesis using Gram matrices or
Guided Correspondence, respectively. Use `--help` parameter to display a list 
of available options. Here are a few examples:

```bash
# Synthesize from data/peebles.jpg and store the output at images/out.jpg (224x224)
python main.py

# Synthesize with Guided Correspondence.
python main-gc.py --input wood.jpg --output images/wood-large.jpg --output_size 512 512

# Reproduce the biased synthesize experiment without occurrence penalty.
# On a 4060Ti this runs in 40 seconds.
python main-gc.py \
    --input data/rust2-small.jpg \
    --output images/rust2-biased.jpg \
    --output_size 384 384 \
    --scales 1.0 \
    --epochs 400 \
    --coef_occur 0.0 \
    --color_bias 0.8104 -0.0924 -0.0964

# View the noise used for the previous experiment.
python main-gc.py \
    --input data/rust2-small.jpg \
    --output images/rust2-biased.jpg \
    --output_size 384 384 \
    --scales 1.0 \
    --epochs 0 \
    --coef_occur 0.0 \
    --color_bias 0.8104 -0.0924 -0.0964

```

## Performance
`--bf16` option can be used to accelerate VGG inference. The option can be used
with the Gram matrix method, but is not favored for the Guided Correspondence
method.

We default to using FP16 for calculating the patch similarity and enabling
`torch.compile` for the loss calculation, which will significantly accelerate
the synthesis pipeline. Currently we did not provide flags to remove it. Feel
free to remove if your device does not support these features.

If `torch.compile` fails to find some CUDA headers, in a conda environment you
may run `conda install cuda-toolkit`.
