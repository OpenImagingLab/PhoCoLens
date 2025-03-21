python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py with $1 distdataparallel=True -p
# python val.py with $1 -p
# python  train.py with $1 -p