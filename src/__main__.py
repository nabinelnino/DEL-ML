
from utils.config_parser import MLConfigParser
from utils.data_reader import DataReader
from src.train import Model
import time
import numpy as np
import sys
import argparse


def main(config_file: str):
    t1 = time.time()
    config = MLConfigParser(config_file, "ml_config")
    config_dict = config.get_config()
    training_cols = config_dict.get("columns_of_interest")
    label_col = config_dict.get("target_col")
    is_binary = config_dict.get("is_binarized_data")
    model_name = config_dict.get("model_name")
    dataset_location = config_dict.get("input_data_path")
    config_location = config_dict.get("processed_file_location")
    result_output = config_dict.get("result_output")
    smile_location = config_dict.get("smile_location")
    isdry_run = config_dict.get("isdry_run", True)
    model_directory = config_dict.get("model_save_directory", "/app/models/")
    model_file_path = f"{model_directory}/{model_name}.pkl"

    reader = DataReader()

    X, y = reader._read_data(file_path=dataset_location, fps=training_cols,
                             label=label_col, model_name=model_name, config_file_path=config_location, binarize=is_binary, dry_run=isdry_run)

    for _, fp_val in X.items():
        clusters = reader.cluster_leader_from_array(fp_val)
    model = Model(model_file_path)
    model.cv(train_data=X, binary_labels=y, source=dataset_location,
             model_name=model_name, config_path=config_location, clusters=clusters)
    model.fit(train_data=X, binary_labels=y, source=dataset_location,
              model_name=model_name, config_path=config_location, clusters=clusters,)

    t2 = time.time()

    print("Total process time", t2-t1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/default_config.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()
    main(args.config)

# Total process time 1465.4664180278778 using vertex ai core 8
# python -m src --config ./config/ml_config.yaml
