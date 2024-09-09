# SATEER: Subject-Aware Transformer for EEG-based Emotion Recognition

This is a PyTorch implementation of SATEER, proposed in our paper "SATEER: Subject-Aware Transformer for EEG-based Emotion Recognition".



## Environment setup
- Install [Anaconda](https://www.anaconda.com/download)

- Create a new `conda` environment:
    ```bash
    conda create --name sateer python=3.8 && conda activate sateer
    ```
- Install the requirements:
    ```bash
    pip install -r requirements.txt
    ```
And that's it, ready to go!

## Datasets 
Download and place the datasets in a folder.

| Dataset name | Download  |
|--------------|-----------|
| DEAP         | [link](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html) |
| DREAMER      | [link](https://zenodo.org/records/546113) |
| AMIGOS       | [link](https://www.eecs.qmul.ac.uk/mmv/datasets/amigos/download.html) |
| SEED         | [link](https://bcmi.sjtu.edu.cn/home/seed/seed.html) |


## Training a model
To train a model, simply run `train.py`: 
```bash
python train.py <DATASET_NAME> <DATASET_PATH> --checkpoints_path=<CHECKPOINTS_PATH> --seed=42 --batch_size=256 --windows_size=1 --windows_stride=1
```

For example, this is the command to train a model on the DEAP dataset, located at `../datasets/deap`, and save the checkpoints to `../checkpoints/deap_model`:
```bash
python train.py deap ../datasets/deap --checkpoints_path=../checkpoints/deap_model --seed=42 --batch_size=256 --windows_size=1 --windows_stride=1
```

The description of the parameters accepted by the script can be viewed using `python train.py -h`.

## Comparison with SOTA
TODO