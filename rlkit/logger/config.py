import os


# Used by wandb
PROJECT_NAME = "dydiff"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOCAL_LOG_DIR = os.path.join(BASE_DIR, "logs")