DATASET='semantickitti'
SEQUENCE='08'
START=0
END=-1
STEP=10
CHUNK_SIZE=20
PARAMDIR='../../config/ground_detection/'

python ground_detection.py --dataset=$DATASET --method='qfz' --params=$PARAMDIR'lambda_qfz.yaml' --sequence=$SEQUENCE --start=$START --end=$END --step=$STEP --chunk_size=$CHUNK_SIZE
python ground_detection.py --dataset=$DATASET --method='hybrid' --params=$PARAMDIR'hybrid_eig.yaml' --sequence=$SEQUENCE --start=$START --end=$END --step=$STEP --chunk_size=$CHUNK_SIZE
python ground_detection.py --dataset=$DATASET --method='ransac' --params=$PARAMDIR'ransac.yaml' --sequence=$SEQUENCE --start=$START --end=$END --step=$STEP --chunk_size=$CHUNK_SIZE
python ground_detection.py --dataset=$DATASET --method='csf' --params=$PARAMDIR'csf_avg_all.yaml' --sequence=$SEQUENCE --start=$START --end=$END --step=$STEP --chunk_size=$CHUNK_SIZE
python ground_detection.py --dataset=$DATASET --method='csf' --params=$PARAMDIR'csf_avg_mask.yaml' --sequence=$SEQUENCE --start=$START --end=$END --step=$STEP --chunk_size=$CHUNK_SIZE
python ground_detection.py --dataset=$DATASET --method='cnn' --params=$PARAMDIR'cnn.yaml' --sequence=$SEQUENCE --start=$START --end=$END --step=$STEP --chunk_size=$CHUNK_SIZE