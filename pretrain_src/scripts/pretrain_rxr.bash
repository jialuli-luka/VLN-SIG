NODE_RANK=0
NUM_GPUS=2
CUDA_VISIBLE_DEVICES='1,0' python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK \
    pretrain_src/main_r2r.py --world_size ${NUM_GPUS} \
    --model_config pretrain_src/config/rxr_model_config.json \
    --config pretrain_src/config/pretrain_rxr.json \
    --output_dir datasets/RXR/exprs/pretrain_new/agent
