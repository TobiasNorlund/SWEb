# Setup environment for running SWEb

## Create conda environment

```bash
conda create -n SWEb python=3.10
conda activate SWEb
```

## Setup CCNet submodule

Required for running the `CCNet` task

```bash
# Install CC-Net submodule
cd cc_net_repo
git submodule init
git submodule update

# Install CC-Net (including pip package) and download/train artifacts
make install
make langs=no,da,is dl_all_lms
make lang=sv lm  # No pretrained model available for swedish, so we need to train one
```

## Install pipeline dependencies

```bash
# From repo root
pip install -r requirements.txt
```

## Setup Dask config

Required for the `ProcessMarkdown` task

```bash
# Create config file for dask (this is needed for torch DataLoaders to spawn processes from within dask workers)
cat << EOF > ~/.config/dask/conf.yaml
distributed:
  worker:
    daemon: False
EOF
```

## Download/install pandoc

Required for `ProcessMarkdown` task

```bash
wget https://github.com/jgm/pandoc/releases/download/2.9.2.1/pandoc-2.9.2.1-linux-amd64.tar.gz -P /tmp
tar -xzf /tmp/pandoc-2.9.2.1-linux-amd64.tar.gz -C $HOME/bin --strip-components=2 pandoc-2.9.2.1/bin/pandoc
```

## Download language identification model

Required for `FilterDocuments`

```bash
mkdir pipeline/bin
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -O pipeline/bin/lid.176.bin
```
