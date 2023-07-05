import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def setup_logging(name, filename=LOG_FILE_PATH):
    logging.basicConfig(
        filename=LOG_FILE_PATH,
        level=logging.INFO,
    )
    return logging.getLogger(name)


if __name__ == "__main__":
    logging.info(
        "Logging works perfectly, remove the if__name__ once you see the this log"
    )
