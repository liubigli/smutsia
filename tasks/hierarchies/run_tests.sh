CKPTPATH='/home/leonardo/Dev/github/smutsia/ckpt/hierachies/toydatasets'
TSAMPLE=200
LOGDIR='/home/leonardo/Dev/github/smutsia/logs/hierarchies/toydatasets'
METRIC='cosine'

python test_hyperbolic.py --logdir=$LOGDIR --path=$BASEPATH'/moons/cosine/mlp_no_noise/' --data='moons' --test_samples=$TSAMPLE
python test_hyperbolic.py --logdir=$LOGDIR --path=$BASEPATH'/moons/cosine/mlp_noise/' --data='moons' --test_samples=$TSAMPLE
python test_hyperbolic.py --logdir=$LOGDIR --path=$BASEPATH'/moons/cosine/dgcnn_no_noise/' --data='moons' --test_samples=$TSAMPLE
python test_hyperbolic.py --logdir=$LOGDIR --path=$BASEPATH'/moons/cosine/dgcnn_noise/' --data='moons' --test_samples=$TSAMPLE

python test_hyperbolic.py --logdir=$LOGDIR --path=$BASEPATH'/circles/cosine/mlp_no_noise/' --data='circles' --test_samples=$TSAMPLE
python test_hyperbolic.py --logdir=$LOGDIR --path=$BASEPATH'/circles/cosine/mlp_noise/' --data='circles' --test_samples=$TSAMPLE
python test_hyperbolic.py --logdir=$LOGDIR --path=$BASEPATH'/circles/cosine/dgcnn_no_noise/' --data='circles' --test_samples=$TSAMPLE
python test_hyperbolic.py --logdir=$LOGDIR --path=$BASEPATH'/circles/cosine/dgcnn_noise/' --data='circles' --test_samples=$TSAMPLE

python test_hyperbolic.py --logdir=$LOGDIR --path=$BASEPATH'/blobs/cosine/mlp_no_noise/' --data='blobs' --num_blobs=9 --test_samples=$TSAMPLE
python test_hyperbolic.py --logdir=$LOGDIR --path=$BASEPATH'/blobs/cosine/mlp_noise/' --data='blobs' --num_blobs=9 --test_samples=$TSAMPLE
python test_hyperbolic.py --logdir=$LOGDIR --path=$BASEPATH'/blobs/cosine/dgcnn_no_noise/' --data='blobs' --num_blobs=9 --test_samples=$TSAMPLE
python test_hyperbolic.py --logdir=$LOGDIR --path=$BASEPATH'/blobs/cosine/dgcnn_noise/' --data='blobs' --num_blobs=9 --test_samples=$TSAMPLE

python test_hyperbolic.py --logdir=$LOGDIR --path=$BASEPATH'/aniso/cosine/mlp_no_noise/' --data='aniso' --num_blobs=9 --test_samples=$TSAMPLE
python test_hyperbolic.py --logdir=$LOGDIR --path=$BASEPATH'/aniso/cosine/mlp_noise/' --data='aniso' --num_blobs=9 --test_samples=$TSAMPLE
python test_hyperbolic.py --logdir=$LOGDIR --path=$BASEPATH'/aniso/cosine/dgcnn_no_noise/' --data='aniso' --num_blobs=9 --test_samples=$TSAMPLE
python test_hyperbolic.py --logdir=$LOGDIR --path=$BASEPATH'/aniso/cosine/dgcnn_noise/' --data='aniso' --num_blobs=9 --test_samples=$TSAMPLE

python test_hyperbolic.py --logdir=$LOGDIR --path=$BASEPATH'/varied/cosine/mlp/' --data='varied' --num_blobs=9 --test_samples=$TSAMPLE
python test_hyperbolic.py --logdir=$LOGDIR --path=$BASEPATH'/varied/cosine/dgcnn/' --data='varied' --num_blobs=9 --test_samples=$TSAMPLE


