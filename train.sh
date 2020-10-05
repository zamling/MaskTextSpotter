/home/zem/.conda/envs/zemPy/bin/python -m torch.distributed.launch --nproc_per_node=2 tools/train_net.py --config-file configs/pretrain.yaml #
# python3 -m torch.distributed.launch --nproc_per_node=8 tools/train_net.py --config-file configs/finetune.yaml #
#CUDA_VISIBLE_DEVICES="1"