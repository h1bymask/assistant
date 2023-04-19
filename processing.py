from pathlib import Path

import numpy as np
from utils.aggregation import aggregate_data
from utils.calculate_features import load_features
import hydra
from os import listdir
from os.path import isfile, join


@hydra.main(config_path="conf", config_name="config")
def processing(cfg) -> None:
    """
    processing raw data for training
    """
    threshold = cfg.threshold
    if threshold > 1 or threshold < 0:
        raise AttributeError

    np.seterr(divide="ignore")

    wavs_folder = cfg.wavs_folder
    features_folder = Path(cfg.features_folder)

    wavs_names = [f for f in listdir(wavs_folder) if isfile(join(wavs_folder, f))]
    load_features(
        wavs_path=wavs_folder,
        wavs_names=wavs_names,
        result_dir=features_folder,
        dataset_name="inference data",
        recalculate_feature=True,
    )

    aggregate_data(features_folder.parent, features_folder, use_tsv=False, threshold=threshold)


if __name__ == "__main__":
    processing()  # pylint: disable=no-value-for-parameter
