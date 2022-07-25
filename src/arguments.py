import argparse
import os

import torch


project_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(project_dir, 'data')


def parse_args():
    parser = argparse.ArgumentParser(description='Finetune a transformers model on a text classification task')
    parser.add_argument(
        '--train_file', type=str, default=None, help='A csv or a json file containing the training data.'
    )
    parser.add_argument(
        '--validation_file', type=str, default=None, help='A csv or a json file containing the validation data.'
    )
    parser.add_argument(
        '--test_file', type=str, default=None, help='A csv or a json file containing the validation data.'
    )
    parser.add_argument('--task_dataset',
                        type=str,
                        default=None,
                        help='dataset name',
                        choices=['kornli', 'kornli-efl', 'ksc', 'nsmc'],
                        )
    parser.add_argument("--negative_num",
                        default=1,
                        type=int,
                        help="Random negative sample number for efl strategy")
    parser.add_argument(
        '--max_length',
        type=int,
        default=256,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        '--pad_to_max_length',
        action='store_true',
        help='If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.',
    )
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        '--vocab_path',
        type=str,
        help="Path to pretrained tokenizer vocab",
    )
    parser.add_argument(
        '--use_slow_tokenizer',
        action='store_true',
        help='If passed, will use a slow tokenizer (not backed by the Tokenizers library).',
    )
    parser.add_argument(
        '--per_device_train_batch_size',
        type=int,
        default=16,
        help='Batch size (per device) for the training dataloader.',
    )
    parser.add_argument(
        '--per_device_eval_batch_size',
        type=int,
        default=128,
        help='Batch size (per device) for the evaluation dataloader.',
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help="initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay to use.')
    parser.add_argument('--num_train_epochs', type=int, default=3, help='Total number of training epochs to perform.')
    parser.add_argument(
        '--max_train_steps',
        type=int,
        default=None,
        help='Total number of training steps to perform. If provided, overrides num_train_epochs.',
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help='Number of updates steps to accumulate before performing a backward/update pass.',
    )
    parser.add_argument(
        '--lr_scheduler_type',
        type=str,
        default='linear',
        help='The scheduler type to use.',
        choices=['linear', 'ReduceLROnPlateau', 'CosineAnnealingLR'],
    )
    parser.add_argument(
        '--use_fp16',
        action='store_true',
        help='Number of updates steps to accumulate before performing a backward/update pass.',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='using cpu or gpu',
    )
    parser.add_argument(
        '--patience', type=int, default=1, help='Number of patience for ReduceLROnPlateau lr_scheduler'
    )
    parser.add_argument(
        '--num_warmup_steps', type=int, default=0, help='Number of steps for the warmup in the lr scheduler.'
    )
    parser.add_argument('--output_dir', type=str, default=None, help='Where to store the final model.')
    parser.add_argument('--seed', type=int, default=None, help='A seed for reproducible training.')
    parser.add_argument(
        '--checkpointing_steps',
        type=str,
        default=None,
        help='Whether the various states should be saved at the end of every n steps, or "epoch" for each epoch.',
    )
    parser.add_argument(
        "--rdrop_coef",
        default=0.0,
        type=float,
        help=
        "The coefficient of KL-Divergence loss in R-Drop paper, for more detail please refer to "
        "https://arxiv.org/abs/2106.14448), if rdrop_coef > 0 then R-Drop works"
    )
    parser.add_argument(
        '--resume_from_checkpoint',
        type=str,
        default=None,
        help='If the training should continue from a checkpoint folder. start from project folder',
    )
    parser.add_argument(
        '--with_tracking',
        action='store_true',
        help='Whether to enable experiment trackers for logging.',
    )
    parser.add_argument(
        '--report_to',
        type=str,
        default='all',
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument('--project_name', type=str, default='EFL_implementation', help='wandb project name')
    parser.add_argument('--run_name', type=str, default='test', help='wandb run name')
    parser.add_argument('--entity', type=str, default=None, help='wandb entity')
    parser.add_argument(
        '--ignore_mismatched_sizes',
        action='store_true',
        help='Whether or not to enable to load a pretrained model whose head dimensions are different.',
    )
    args = parser.parse_args()

    # output_dir 재정의
    args.output_dir = os.path.join(project_dir, args.output_dir)
    # TBERT 사용시
    if os.path.isdir(os.path.join(project_dir, args.model_name_or_path)):
        args.model_name_or_path = os.path.join(project_dir, args.model_name_or_path)
    if args.vocab_path:
        if os.path.isdir(os.path.join(project_dir, args.vocab_path)):
            args.vocab_path = os.path.join(project_dir, args.vocab_path)
    else:
        args.vocab_path = args.model_name_or_path

    # resume_from_checkpoint 사용시
    if args.resume_from_checkpoint is not None:
        args.resume_from_checkpoint = os.path.join(project_dir, args.resume_from_checkpoint)

    # Sanity checks
    if args.train_file is None and args.validation_file is None:
        raise ValueError("Need a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split('.')[-1]
            assert extension in ['csv', 'json'], '`train_file` should be a csv or a json file.'
        if args.validation_file is not None:
            extension = args.validation_file.split('.')[-1]
            assert extension in ['csv', 'json'], '`validation_file` should be a csv a json file.'

    return args
