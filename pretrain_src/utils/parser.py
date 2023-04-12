import argparse
import sys
import json


def load_parser():
    parser = argparse.ArgumentParser()

    # Required parameters
    # NOTE: train tasks and val tasks cannot take command line arguments
    parser.add_argument(
        "--model_config", type=str, help="path to model structure config json"
    )
    parser.add_argument(
        "--checkpoint", default=None, type=str, help="path to model checkpoint (*.pt)"
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="The output directory where the model checkpoints will be written.",
    )

    # training parameters
    parser.add_argument(
        "--train_batch_size",
        default=4096,
        type=int,
        help="Total batch size for training. ",
    )
    parser.add_argument(
        "--val_batch_size",
        default=4096,
        type=int,
        help="Total batch size for validation. ",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Number of updates steps to accumualte before "
        "performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=3e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--valid_steps", default=1000, type=int, help="Run validation every X steps"
    )
    parser.add_argument("--log_steps", default=1000, type=int)
    parser.add_argument(
        "--num_train_steps",
        default=100000,
        type=int,
        help="Total number of training updates to perform.",
    )
    parser.add_argument(
        "--optim",
        default="adamw",
        choices=["adam", "adamax", "adamw"],
        help="optimizer",
    )
    parser.add_argument(
        "--betas", default=[0.9, 0.98], nargs="+", help="beta for adam optimizer"
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="tune dropout regularization"
    )
    parser.add_argument(
        "--weight_decay",
        default=0.01,
        type=float,
        help="weight decay (L2) regularization",
    )
    parser.add_argument(
        "--grad_norm",
        default=2.0,
        type=float,
        help="gradient clipping (-1 for no clipping)",
    )
    parser.add_argument(
        "--warmup_steps",
        default=10000,
        type=int,
        help="Number of training steps to perform linear " "learning rate warmup for.",
    )

    # device parameters
    parser.add_argument(
        "--seed", type=int, default=0, help="random seed for initialization"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit float precision instead of 32-bit",
    )
    parser.add_argument(
        "--n_workers", type=int, default=4, help="number of data workers"
    )
    parser.add_argument("--pin_mem", action="store_true", help="pin memory")

    # distributed computing
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank for distributed training on gpus",
    )
    parser.add_argument(
        "--node_rank",
        type=int,
        default=0,
        help="Id of the node",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Number of GPUs across all nodes",
    )

    # can use config files
    parser.add_argument("--config", required=True, help="JSON config files")

    add_beit_argument(parser)

    return parser


def add_beit_argument(parser):
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_false', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=True)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=False)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float,
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument('--num_mask_patches', default=75, type=int,
                        help='number of the visual tokens/patches need be masked')
    parser.add_argument('--max_mask_patches_per_block', type=int, default=None)
    parser.add_argument('--min_mask_patches_per_block', type=int, default=16)

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--second_input_size', default=112, type=int,
                        help='images input size for discrete vae')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)

    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=3.0, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    # parser.add_argument('--weight_decay', type=float, default=0.05,
    #                     help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=1.5e-3, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--layer_decay', type=float, default=0.9)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    # parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
    #                     help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--second_interpolation', type=str, default='lanczos',
                        help='Interpolation for discrete vae (random, bilinear, bicubic default: "lanczos")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')
    parser.add_argument('--disable_weight_decay_on_rel_pos_bias', action='store_true', default=False)

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=0, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', default=False, action='store_true')

    parser.add_argument('--data_set', default='IMNET', choices=['CIFAR', 'IMNET', 'image_folder'],
                        type=str, help='ImageNet dataset path')
    # parser.add_argument('--output_dir', default='',
    #                     help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    # parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    # parser.add_argument('--pin_mem', action='store_true',
    #                     help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--enable_deepspeed', action='store_true', default=False)



def parse_with_config(parser):
    args = parser.parse_args()
    if args.config is not None:
        config_args = json.load(open(args.config))
        override_keys = {
            arg[2:].split("=")[0] for arg in sys.argv[1:] if arg.startswith("--")
        }
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)
    del args.config
    return args
