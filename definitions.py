import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# DATA PATH
DATA_ROOT_PATH = os.path.join(PROJECT_ROOT, 'data')
SEMANTICKITTI_PATH = os.path.join(DATA_ROOT_PATH, 'semantickitti', 'dataset', 'sequences')
SEMANTICKITTI_CONFIG = os.path.join(PROJECT_ROOT, 'config', 'labels', 'semantic-kitti.yaml')
IMAGES_PATH = os.path.join(DATA_ROOT_PATH, 'images',)
MODEL_NET = os.path.join(DATA_ROOT_PATH, 'modelnet')
# RESULTS PATH
RESULTS_ROOT_PATH = os.path.join(PROJECT_ROOT, 'results')
BEA_RESULTS = os.path.join(RESULTS_ROOT_PATH, 'ground_detection', 'lambda_fz', 'projections')

# DEEP LEARNING LOG DIR
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
GROUND_DETECTION_LOG_DIR = os.path.join('ground_detection')