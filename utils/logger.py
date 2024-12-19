import logging
import os
from datetime import datetime

from google.cloud import storage

# GCS Bucket Configuration
GCS_BUCKET_NAME = "test-aircheck"
GCS_LOGS_FOLDER = "logs/"

service_account_path = '../app/service_account.json'

# Check if the file exists before setting the environment variable
if os.path.exists(service_account_path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_path
    print("Service account credentials set.")
    print("service account path---", service_account_path)
else:
    print("Service account file not found. Skipping credential setup.")


LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)
# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler(LOG_FILE_PATH)
file_handler.setFormatter(logging.Formatter(
    "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(console_handler)


def upload_log_to_gcs():
    try:
        # Initialize GCS client
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET_NAME)

        # Upload the log file
        blob = bucket.blob(f"{GCS_LOGS_FOLDER}{LOG_FILE}")
        blob.upload_from_filename(LOG_FILE_PATH)
        logger.info(f"Log file {LOG_FILE} uploaded to GCS bucket \
                    {GCS_BUCKET_NAME}/{GCS_LOGS_FOLDER}")
    except Exception as e:
        logger.error(f"Failed to upload log file to GCS: {e}")
