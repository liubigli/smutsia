DATASET='semantickitti'
SEQUENCE='08'
START=0
END=-1
STEP=100
CHUNK_SIZE=20
PARAMDIR='../../config/ground_detection/'

python ground_detection.py --dataset=$DATASET --method='qfz' --params=$PARAMDIR'lambda_qfz.yaml' --sequence=$SEQUENCE --start=$START --end=$END --step=$STEP --chunk_size=$CHUNK_SIZE
python ground_detection.py --dataset=$DATASET --method='hybrid' --params=$PARAMDIR'hybrid.yaml' --sequence=$SEQUENCE --start=$START --end=$END --step=$STEP --chunk_size=$CHUNK_SIZE
python ground_detection.py --dataset=$DATASET --method='ransac' --params=$PARAMDIR'ransac.yaml' --sequence=$SEQUENCE --start=$START --end=$END --step=$STEP --chunk_size=$CHUNK_SIZE