from pathlib import Path
import shutil

import torch
import numpy as np
import os
from utils.aggregation import aggregate_data
from utils.calculate_features import load_features
import hydra
from os import listdir
from hydra.utils import instantiate
from os.path import isfile, join
from core.model import LABEL2EMO
from rich.console import Console
console = Console()

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
    shutil.rmtree(features_folder / "features")
    Path(features_folder / "features").mkdir(exist_ok=True)

    wavs_names = [f for f in listdir(wavs_folder) if isfile(join(wavs_folder, f))]
    load_features(
        wavs_path=Path(wavs_folder),
        wavs_names=wavs_names,
        result_dir=Path(features_folder),
        dataset_name="inference data",
        recalculate_feature=True,
    )

    names_to_id = {}
    test_samples = []
    for idx, file in enumerate((Path(features_folder) / "features").glob("*")):
        test_samples.append(np.load(file))
        names_to_id[idx] = file.stem
    model = instantiate(cfg.model)
    map_location = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(
        torch.load(
            cfg.best_model_folder + "/" + cfg.model["_target_"].split(".")[-1] + ".pt",
            map_location=torch.device(map_location),
        )
    )
    model.eval()
    with torch.no_grad():
        for idx, feature in enumerate(test_samples):
            prediction = torch.nn.functional.softmax(model(torch.tensor(feature[None, :])))[0]
            output_label = torch.argmax(prediction).item()
            console.print(
                f"file [green]{names_to_id[idx]}[/green]: prediction: [red]{LABEL2EMO[output_label]}[/red] probability: {prediction[output_label]:.4f} ({prediction})"
            )


if __name__ == "__main__":
    processing()  # pylint: disable=no-value-for-parameter
