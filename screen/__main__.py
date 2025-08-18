
from utils.config_parser import MLConfigParser
from screen.screen import Screen
import argparse
import sys
import time
if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/default_config.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()
    t1 = time.time()
    config_file = args.config
    config = MLConfigParser(config_file, "ml_config")
    config_dict = config.get_config()
    training_cols = config_dict.get("columns_of_interest")
    label_col = config_dict.get("target_col")
    is_binary = config_dict.get("is_binarized_data")
    is_groupped_cluster = config_dict.get("cluster_generation")
    model_name = config_dict.get("model_name")
    dataset_location = config_dict.get("input_data_path")
    config_location = config_dict.get("processed_file_location")
    result_output = config_dict.get("result_output")
    smile_location = config_dict.get("smile_location")
    isdry_run = config_dict.get("isdry_run", True)
    model_directory = config_dict.get("model_save_directory", "/app/models/")
    model_file_path = f"{model_directory}/{model_name}.pkl"
    screen = Screen(model_file_path, training_cols, is_binary)
    screen.screen(smile_location, result_output)
    t2 = time.time()
    print("total processing time---", t2-t1)
