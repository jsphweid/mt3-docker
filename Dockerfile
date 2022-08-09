FROM nvidia/cuda:11.7.0-devel-ubuntu18.04

RUN apt-get update && apt-get install -y libssl-dev openssl wget build-essential zlib1g-dev git libfluidsynth1 libasound2-dev libjack-dev libffi-dev libbz2-dev liblzma-dev libsqlite3-dev ffmpeg
RUN wget https://www.python.org/ftp/python/3.7.13/Python-3.7.13.tgz
RUN tar xzvf Python-3.7.13.tgz && cd Python-3.7.13 && ./configure && make && make install

# NOTE: code to set python3 -> python and install pip stolen from here
# https://github.com/docker-library/python/blob/5effd7e8b667d927768d94872f554d3ba9d57ebf/3.7/buster/Dockerfile
RUN set -eux; \
  for src in idle3 pydoc3 python3 python3-config; do \
  dst="$(echo "$src" | tr -d 3)"; \
  [ -s "/usr/local/bin/$src" ]; \
  [ ! -e "/usr/local/bin/$dst" ]; \
  ln -svT "$src" "/usr/local/bin/$dst"; \
  done

ENV PYTHON_PIP_VERSION 21.1.3
ENV PYTHON_GET_PIP_URL https://github.com/pypa/get-pip/raw/aeca83c7ba7f9cdfd681103c4dcbf0214f6d742e/public/get-pip.py

RUN set -eux; \
  \
  wget -O get-pip.py "$PYTHON_GET_PIP_URL"; \
  export PYTHONDONTWRITEBYTECODE=1; \
  \
  python get-pip.py \
  --disable-pip-version-check \
  --no-cache-dir \
  --no-compile \
  "pip==$PYTHON_PIP_VERSION" \
  ; \
  rm -f get-pip.py; \
  \
  pip --version

# create a non-root user
RUN useradd -ms /bin/bash mt3user
USER mt3user
WORKDIR /home/mt3user

# install gsutil, this will be useful for download the model checkpoints later
# Downloading gcloud package
RUN wget -O /tmp/google-cloud-sdk.tar.gz https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz
# Installing the package
RUN mkdir -p /home/mt3user/gcloud \
  && tar -C /home/mt3user/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && /home/mt3user/gcloud/google-cloud-sdk/install.sh
# Adding the package path to mt3user
ENV PATH $PATH:/home/mt3user/gcloud/google-cloud-sdk/bin

# NOTE: much of the next code stolen from magenta
# https://github.com/magenta/mt3/blob/main/mt3/colab/music_transcription_with_transformers.ipynb

# pin CLU for python 3.7 compatibility
# pin Orbax to use Checkpointer
RUN pip install pyfluidsynth clu==0.0.7 orbax==0.0.2

# install t5x
RUN git clone --branch=main https://github.com/google-research/t5x; \
  # pin T5X for python 3.7 compatibility
  cd t5x; \
  git reset --hard 2e05ad41778c25521738418de805757bf2e41e9e; \
  cd ..; \
  mv t5x t5x_tmp; \
  mv t5x_tmp/* .; \
  rm -r t5x_tmp; \
  sed -i 's:jax\[tpu\]:jax:' setup.py; \
  python -m pip install -e .

RUN git clone --branch=main https://github.com/magenta/mt3; \
  mv mt3 mt3_tmp; \
  mv mt3_tmp/* .; \
  rm -r mt3_tmp; \
  python3 -m pip install -e .

RUN pip install --upgrade jax==0.3.15
RUN pip install --upgrade https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.3.15+cuda11.cudnn805-cp37-none-manylinux2014_x86_64.whl

RUN gsutil -q -m cp -r gs://mt3/checkpoints .
COPY ismir2021.gin /home/mt3user/
COPY mt3.gin /home/mt3user/
COPY model.gin /home/mt3user/
COPY app.py /home/mt3user/app.py

RUN pip install flask
RUN apt-get update && apt-get install -y gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:80", "app:app"]