export CUDA_VISIBLE_DEVICES=1
export NGPUS=1
export OMP_NUM_THREADS=2 # you can change this value according to your number of cpu cores

python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port 29102 train.py ../configs/raildb.py