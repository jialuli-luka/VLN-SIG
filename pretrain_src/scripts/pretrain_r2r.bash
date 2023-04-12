NODE_RANK=0
NUM_GPUS=2
CUDA_VISIBLE_DEVICES='2,3' python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK \
    pretrain_src/main_r2r.py --world_size ${NUM_GPUS} \
    --model_config pretrain_src/config/r2r_model_config.json \
    --config pretrain_src/config/pretrain_r2r.json \
    --output_dir datasets/R2R/exprs/pretrain_new/agent
