# mt3-docker

Useful if you want to run the [MT3](https://github.com/magenta/mt3) model on your ubuntu dual-boot gaming rig instead of a colab notebook.

Thanks to the Magenta team for this amazing model.

NOTE: this Dockerfile assumes you:
  1. have a host machine with an eligible GPU
  2. [compatible versions of cuda toolkit and nvidia drivers installed](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) - Dockerfile uses `nvidia/cuda:11.7.0`
  3. [cudnn installed](https://github.com/google/jax)

NOTE: I'm not an expert here and there may be steps here or steps in the Dockerfile that are not necessary. Feel free to make an MR if you see something excessive or incorrect!

### Usage
Once machine is setup properly (see above about setting up GPU), then just build and run the image...
```bash
sudo docker build -t mt3 .
sudo docker run -p 5000:5000 --gpus all mt3 # NOTE: you may need to adjust memory i.e. "-m 12000m"
```

Once that's running, you can simply issue a POST request to `http://<container-ip>:5000/transcribe-anything` (or `http://<container-ip>:5000/transcribe-piano`) with POST data like `{"data": "<base64file16ksamplerate>"}`

Using a 3090 it should take less than a minute for most music files.

### TODO

Dockerfile is really messy. 

 [ ] It's not that efficient or production-ready... calls for model inference could probably benefit from batching I'd think. The model itself can probably be optimized in some way. This is not my wheelhouse but I'm trying to learn more
 [ ] Ideally it shouldn't pull from `devel` I think. I did this so the `ptxas` binary is available (I think jax uses it) but there's probably a better way
 [ ] python/pip installation can probably be greatly simplified
 [ ] t5x/mt3 installation was copied from [colab](https://github.com/magenta/mt3/blob/main/mt3/colab/music_transcription_with_transformers.ipynb) without much thought but it can probably be simplified a bit if we're only targeting GPUs.
 
