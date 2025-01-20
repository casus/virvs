# A Benchmark for Virus Infection Reporter Virtual Staining in Fluorescence and Brightfield Microscopy

This code implements the code for A Benchmark for Virus Infection Reporter Virtual Staining in Fluorescence and Brightfield Microscopy paper.

## How to cite us
Wyrzykowska, Maria, Gabriel della Maggiora, Nikita Deshpande, Ashkan Mokarian, and Artur Yakimovich. "A Benchmark for Virus Infection Reporter Virtual Staining in Fluorescence and Brightfield Microscopy." bioRxiv (2024): 2024-08.

```
@article{wyrzykowska2024benchmark,
  title={A Benchmark for Virus Infection Reporter Virtual Staining in Fluorescence and Brightfield Microscopy},
  author={Wyrzykowska, Maria and della Maggiora, Gabriel and Deshpande, Nikita and Mokarian, Ashkan and Yakimovich, Artur},
  journal={bioRxiv},
  pages={2024--08},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```

## Where to get the data?

The datasets that we are using are available online. They can be downloaded following the instructions in the papers publishing them, which are referenced in our paper.
After downloading the data, run the data processing scripts that are located in `scripts/data_processing` directory, first modifying the paths in them:
- for HAdV data: `preprocess_hadv.py`
- for VACV data: `stitch_vacv.py` + `preprocess_vacv.py`
- for the rest of the viruses: `preprocess_other.py`
- to prepare Cellpose predictions: `prepare_cellpose_preds.py` (for cells) and `prepare_cellpose_preds_nuc.py` (for nuclei)

## How to prepare environment?

Run the following code:
```
conda create -n myenv python=3.10
conda activate myenv
pip install -r requirements.txt
pip install -e .
```

## How to run the training code?

1. Download the data. 
1. Modify the config in `configs/` directory with the path to the data you want to use and the directory for outputs.
2. Run the code from the root directory: `python scripts/training/train_pix2pix.py --config-path $PATH_TO_CONFIG --neptune-token $NEPTUNE_TOKEN` or `python scripts/training/train_unet.py --config-path $PATH_TO_CONFIG --neptune-token $NEPTUNE_TOKEN` .

`--neptune-token` argument is optional.

## How to evaluate the models?

There are multiple scripts available for evaluation:
- testing/evalute.py - to calculate basic metrics such as MSE for all the models and datasets
- testing/evalute_hadv.py - to calculate IoU, F1 etc for HAdV datasets
- testing/evalute_other.py - to calculate IoU, F1 etc for the rest of datasets

1. Modify the `scripts/testing/evalute.py` with the correct paths.
2. Run the code from the root directory: `scripts/testing/evalute.py`.

## License
This repository is released under the MIT License (refer to the LICENSE file for details).
