ob_type=pano
feedback=sample

features=vit-16-ori
ft_dim=512

ngpus=1
seed=0

outdir=../datasets/R2R/exprs/finetune/agent/

flag="--root_dir ../datasets
      --output_dir ${outdir}

      --dataset r2r

      --vlnbert ${vlnbert}
      --ob_type ${ob_type}

      --world_size ${ngpus}
      --seed ${seed}

      --num_l_layers 9
      --num_x_layers 4

      --hist_enc_pano
      --hist_pano_num_layers 2

      --fix_lang_embedding
      --fix_hist_embedding

      --features ${features}
      --feedback ${feedback}

      --max_action_len 15
      --max_instr_len 60

      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --lr 1e-5
      --iters 300000
      --log_every 2000
      --batch_size 6
      --optim adamW

      --ml_weight 0.15

      --feat_dropout 0.4
      --dropout 0.5"

# train
CUDA_VISIBLE_DEVICES='0' python r2r/main.py $flag  \
      --aug ../datasets/R2R/annotations/prevalent_aug_train_enc.json \
      --bert_ckpt_file "add pt ckpt path" \
      --use_ig \
      --ig_head 8192 \
      --ig_path ../dvae_probs.hdf5 \
      --weighted_token \
      --train_ig 0.5
