import accelerate
import dotenv
import os
from loguru import logger

dotenv.load_dotenv()
GLOBAL_ACCELERATOR = accelerate.Accelerator()

device = GLOBAL_ACCELERATOR.device


def main():
    pass


if __name__ == "__main__":
    main()
