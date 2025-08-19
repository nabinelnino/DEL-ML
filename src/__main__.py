
from utils.config_parser import MLConfigParser
from utils.data_reader import DataReader
from src.train import Model
import time
import sys
import argparse
from datetime import datetime


def main(config_file: str):
    t1 = time.time()
    config = MLConfigParser(config_file, "ml_config")
    config_dict = config.get_config()
    training_cols = config_dict.get("columns_of_interest")
    label_col = config_dict.get("target_col")
    is_binary = config_dict.get("is_binarized_data")
    model_name = config_dict.get("model_name")

    list_of_files = (config_dict.get("input_data_path"))
    input_bucket = config_dict.get("input_data_bucket")
    input_data_folder = config_dict.get("input_data_folder")
    config_location = config_dict.get("processed_file_location")
    result_output = config_dict.get("result_output")
    smile_location = config_dict.get("smile_location")
    partner_name = config_dict.get("partner_name")
    isdry_run = config_dict.get("isdry_run", True)
    model_directory = config_dict.get("model_save_directory", "/app/models/")
    model_file_path = f"{model_directory}/{model_name}.pkl"

    reader = DataReader()

    if input_bucket and input_data_folder:
        print(f"bucket is {input_bucket} and folder is {input_data_folder}")
        list_of_files = reader.list_gcs_files(bucket_name=input_bucket,
                                              folder_path=input_data_folder)
        print("list files-----", list_of_files)
    if not isinstance(list_of_files, list):
        list_of_files = [list_of_files]
    for file in list_of_files:
        dataset_location = file
        model_name = reader.create_model_name(
            dataset_location=dataset_location, partner_name=partner_name)

        start_time = datetime.now()
        print("processing file------", file)

        X, y = reader._read_data(file_path=dataset_location, fps=training_cols,
                                 label=label_col, model_name=model_name, config_file_path=config_location, binarize=is_binary, dry_run=isdry_run, list_of_bucket_files=list_of_files)

        end_time = datetime.now()
        print(f"Total time to read dataset is {end_time-start_time}")
        start_time_cluster = datetime.now()
        for _, fp_val in X.items():
            clusters = reader.cluster_leader_from_array(fp_val)
        end_time_cluster = datetime.now()
        # model_name = "test_experiment"
        print(
            f"time to create cluster is {end_time_cluster - start_time_cluster}")
        start_time_model = datetime.now()
        model = Model(model_file_path)
        model.cv(train_data=X, binary_labels=y, source=dataset_location,
                 model_name=model_name, config_path=config_location, clusters=clusters)
        model.fit(train_data=X, binary_labels=y, source=dataset_location,
                  model_name=model_name, config_path=config_location, clusters=clusters,)
        end_time_model = datetime.now()
        print("total model building time is",
              end_time_model - start_time_model)
        t2 = time.time()

        print("Total process time", t2-t1)
        exit()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/default_config.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()
    print("processing starting form here", args.config)
    main(args.config)
