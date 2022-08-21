# mt3-docker

Useful if you want to run the [MT3](https://github.com/magenta/mt3) model on your ubuntu dual-boot gaming rig instead of a colab notebook.

Thanks to the Magenta team for this amazing model.

I originally made [this docker file](https://github.com/magenta/mt3/issues/70#issuecomment-1207521792) but wanted to share.

NOTE: this Dockerfile assumes you:
  1. have a host machine with an eligible GPU
  2. [compatible versions of cuda toolkit and nvidia drivers installed](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) - Dockerfile uses `nvidia/cuda:11.7.0`
  3. [cudnn installed](https://github.com/google/jax)

NOTE: I'm not an expert here and there may be steps here or steps in the Dockerfile that are not necessary. Feel free to make an MR if you see something excessive or incorrect!

### TODO

Dockerfile is really messy. 

 - Ideally it shouldn't pull from `devel` I think. I did this so the `ptxas` binary is available (I think jax uses it) but there's probably a better way
 - python/pip installation can probably be greatly simplified
 - t5x/mt3 installation was copied from [colab](https://github.com/magenta/mt3/blob/main/mt3/colab/music_transcription_with_transformers.ipynb) without much thought but it can probably be simplified a bit if we're only targeting GPUs.
 
