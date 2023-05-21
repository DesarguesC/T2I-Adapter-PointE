# T2I-Adapter for point-e 3D model

environment variable:
```bat
export RANK=0
export WORLD_SIZE=2     # the number of the thread to be called
export MASTER_ADDR=localhost
export MASTER_PORT=5678
```


multi gpus train command:
```bat
python -m torch.distributed.launch --nproc_per_node=<gpu ammounts> --master_port=5678 train_point.py --ckpt ~/autodl-tmp/models/v1-5-pruned.ckpt --data_path ~/autodl-tmp/COCO/
```