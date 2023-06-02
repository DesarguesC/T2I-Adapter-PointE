# T2I-Adapter for point-e 3D model

environment variable:
```sh
export RANK=0
export WORLD_SIZE=2     # the number of the thread to be called
export MASTER_ADDR=localhost
export MASTER_PORT=5678
export CUDA_VISIBLE_DEVICES=<your gpu num>
```


multi gpus train command:
```python
python -m torch.distributed.launch --nproc_per_node=<gpu ammounts> --master_port=5678 train_point.py --ckpt ~/autodl-tmp/models/v1-5-pruned.ckpt --data_path ~/autodl-tmp/COCO/
```


test
```python
python test_adapter.py --which_cond pointE --cond_path ~/autodl-tmp/COCO/total_data/point_img/0000000000.png.jpg --prompt "A bicycle replica with a clock as the front wheel." --adapter_ckpt experiments/train_point/models/model_ad_40000.pth --sd_ckpt ~/autodl-tmp/models/v1-5-pruned.ckpt --config experiments/train_point/sd-v1-train.yaml --cond_inp_type 'pointE'

```
