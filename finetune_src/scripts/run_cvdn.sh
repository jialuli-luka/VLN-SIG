
#features=vitbase_r2rfte2e
features=vit-16-ori
ft_dim=512

ngpus=1
seed=0

outdir=../datasets/CVDN/exprs/finetune/agent/

flag="--root_dir ../datasets
      --output_dir ${outdir}

      --dataset cvdn
      --use_player_path

      --ob_type pano

      --world_size ${ngpus}
      --seed ${seed}

      --num_l_layers 9
      --num_x_layers 4

      --hist_enc_pano
      --hist_pano_num_layers 2

      --no_lang_ca

      --features ${features}
      --feedback sample

      --max_action_len 30
      --max_instr_len 100

      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --lr 1e-5
      --iters 200000
      --log_every 1000
      --batch_size 8
      --optim adamW

      --ml_weight 0.15

      --feat_dropout 0.4
      --dropout 0.5"


# train
CUDA_VISIBLE_DEVICES='0' python cvdn/main.py $flag \
      --bert_ckpt_file "add pt ckpt here" \
      --eval_first
