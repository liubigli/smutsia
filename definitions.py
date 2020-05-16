import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# DATA PATH
DATA_ROOT_PATH = os.path.join(PROJECT_ROOT, 'data')
SEMANTICKITTI_PATH = os.path.join(DATA_ROOT_PATH, 'semantickitti', 'dataset', 'sequences')

# RESULTS PATH
RESULTS_ROOT_PATH = os.path.join(PROJECT_ROOT, 'results')
BEA_RESULTS = os.path.join(RESULTS_ROOT_PATH, 'ground_detection', 'lambda_fz', 'projections')

