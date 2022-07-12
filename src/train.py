import argparse
import json
import math
import os
import sys
import random
import logging

import wandb
import numpy as np
import torch
from datasets import load_from_disk
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from accelerate.utils import set_seed
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    RobertaForSequenceClassification,
    AutoTokenizer,
    BertTokenizer,
    DataCollatorWithPadding,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

project_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(project_dir, 'data')

sys.path.append(project_dir)

from src.utils import save_pickle, load_pickle, AverageMeter, make_dataset

# wandb description silent
os.environ['WANDB_SILENT'] = "true"

# gpu setting
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Set the GPU 0 to use

logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        default=16,
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
        type=SchedulerType,
        default='linear',
        help='The scheduler type to use.',
        choices=['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'],
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
        '--resume_from_checkpoint',
        type=str,
        default=None,
        help='If the training should continue from a checkpoint folder.',
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
    # TRoBERTa 사용시
    if os.path.isdir(os.path.join(project_dir, args.model_name_or_path)):
        args.model_name_or_path = os.path.join(project_dir, args.model_name_or_path)
    if args.vocab_path:
        if os.path.isdir(os.path.join(project_dir, args.vocab_path)):
            args.vocab_path = os.path.join(project_dir, args.vocab_path)

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


def get_dataloader(args, datasets, tokenizer, padding, label_to_id):
    # Preprocessing the datasets
    non_label_column_names = [name for name in datasets['train'].column_names if name != 'label']
    if 'sentence1' in non_label_column_names and 'sentence2' in non_label_column_names:
        sentence1_key, sentence2_key = 'sentence1', 'sentence2'
    else:
        if len(non_label_column_names) >= 2:
            sentence1_key, sentence2_key = non_label_column_names[:2]
        else:
            sentence1_key, sentence2_key = non_label_column_names[0], None

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts,
                           padding=padding,
                           max_length=args.max_length,
                           truncation=True,
                           return_token_type_ids=False,
                           )

        if 'label' in examples:
            # Map labels to IDs
            result['labels'] = [label_to_id[l] for l in examples['label']]

        return result

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done or max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if args.use_fp16 else None))

    dataloader_list = []
    for k in ['train', 'validation', 'test']:
        k_dataset = datasets[k]

        k_dataset = k_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=datasets['train'].column_names,
            desc='Running tokenizer on dataset',
        )

        if k == 'train':
            k_dataloader = DataLoader(k_dataset,
                                      collate_fn=data_collator,
                                      batch_size=args.per_device_train_batch_size,
                                      shuffle=True,
                                      )
        else:
            k_dataloader = DataLoader(k_dataset,
                                      collate_fn=data_collator,
                                      batch_size=args.per_device_train_batch_size,
                                      )

        dataloader_list.append(k_dataloader)

    return dataloader_list


def main():
    args = parse_args()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # Get the datasets
    # Loading the dataset from local csv or json file.
    if not os.path.exists(os.path.join(data_dir, 'kornli_dataset')):
        make_dataset(args)
    raw_datasets = load_from_disk(os.path.join(data_dir, 'kornli_dataset'))

    # Labels
    # Trying to have good defaults here, don't hesitate to tweak to your needs.
    is_regresssion = raw_datasets['train'].features['label'].dtype in ['float32', 'float64']
    if is_regresssion:
        num_labels = 1
    else:
        # A useful fast method
        label_list = raw_datasets['train'].unique('label')
        label_list.sort()  # Let's sort it for determinism
        num_labels = len(label_list)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    if args.vocab_path is not None and os.path.isdir(args.vocab_path):
        tokenizer = BertTokenizer.from_pretrained(args.vocab_path,
                                                  do_lower_case=False,
                                                  unk_token='<unk>',
                                                  sep_token='</s>',
                                                  pad_token='<pad>',
                                                  cls_token='<s>',
                                                  mask_token='<mask>',
                                                  model_max_length=args.max_length)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    if os.path.isdir(args.model_name_or_path):
        model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool('.ckpt' in args.model_name_or_path),
            config=config,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        )

    model.to(args.device)

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = {v: i for i, v in enumerate(label_list)}

    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = 'max_length' if args.pad_to_max_length else False

    train_dataloader, eval_dataloader, test_dataloader = get_dataloader(args, raw_datasets, tokenizer, padding, label_to_id)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        }
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader)) / args.gradient_accumulation_steps
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if hasattr(args.checkpointing_steps, 'isdigit'):
        checkpointing_steps = args.checkpointing_steps
        if args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
    else:
        checkpointing_steps = None

    # We need to initialize the trackers we use, and also store our configuration.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config['lr_scheduler_type'] = experiment_config['lr_scheduler_type'].value
        # wandb initialize
        wandb.init(project=args.project_name,
                   name=args.run_name,
                   entity=args.entity,
                   config=experiment_config,
                   reinit=True)

    # Train!
    total_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps

    logger.info('***** Running training *****')
    logger.info(f'  Num examples = {len(raw_datasets["train"])}')
    logger.info(f'  Num Epochs = {args.num_train_epochs}')
    logger.info(f'  Instantaneous batch size per device = {args.per_device_train_batch_size}')
    logger.info(f'  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}')
    logger.info(f'  Gradient Accumulation steps = {args.gradient_accumulation_steps}')
    logger.info(f'  Total optimization steps = {args.max_train_steps}')

    global_step = 1
    updated_step = 0
    if args.with_tracking:
        train_loss = AverageMeter()
        train_acc = AverageMeter()

    if isinstance(checkpointing_steps, int):
        best_checkpoining_steps = None
        best_valid_loss = float('inf')

    for epoch in range(args.num_train_epochs):
        print(f'\n{epoch + 1} epoch start!')
        torch.cuda.empty_cache()

        train_iterator = tqdm(train_dataloader, desc='train-Iteration')
        for step, batch in enumerate(train_iterator):
            loss, acc, updated_step = training_per_step(args, batch, model, optimizer, criterion, global_step, updated_step)
            train_loss.update(loss, args.per_device_train_batch_size)
            train_acc.update(acc / args.per_device_train_batch_size)
            global_step += 1
            description = f'{epoch + 1}epoch {global_step: >5d}step | loss: {train_loss.avg:.4f} | acc: {train_acc.avg:.4f} | best_loss: {best_valid_loss:.4f}'
            train_iterator.set_description(description)

            if args.with_tracking:
                last_lr = lr_scheduler.get_last_lr()[0]
                wandb.log({
                    'train/loss': train_loss.avg,
                    'train/acc': train_acc.avg,
                    'train/learning_rate': last_lr,
                })

            if isinstance(checkpointing_steps, int):
                if (updated_step + 1) % checkpointing_steps == 0:
                    with torch.no_grad():
                        valid_loss, valid_acc = evaluating(args, eval_dataloader, model, criterion)

                        lr_scheduler.step()

                        if valid_loss < best_valid_loss:
                            best_valid_loss = valid_loss
                            best_checkpoining_steps = updated_step
                            output_dir = f'step_{best_checkpoining_steps}'
                            if args.output_dir is not None:
                                output_dir = os.path.join(args.output_dir, output_dir)
                            model.save_pretrained(output_dir)
                            tokenizer.save_pretrained(output_dir)

                    if args.with_tracking:
                        wandb.log({
                            'train/loss': train_loss.avg,
                            'train/acc': train_acc.avg,
                            'train/learning_rate': last_lr,
                            'eval/best_loss': best_valid_loss,
                            'eval/loss': valid_loss,
                            'eval/acc': valid_acc,
                            'global_step': global_step,
                        })
                        train_loss.reset()
                        train_acc.reset()

            if updated_step >= args.max_train_steps:
                break

    if args.output_dir is not None:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    # load model at best checkpoint step
    model = model.from_pretrained(os.path.join(args.output_dir, f'step_{best_checkpoining_steps}'))

    # Final evaluation on kornli test set
    with torch.no_grad():
        test_loss, test_acc = evaluating(args, test_dataloader, model, criterion)

    logger.info(f'\nxnli testset: accuracy: {test_acc:.4f}')

    if args.output_dir is not None:
        with open(os.path.join(args.output_dir, 'all_results.json'), 'w') as f:
            json.dump({'test_accuracy': test_acc}, f)

    if args.with_tracking:
        wandb.finish()


def training_per_step(args, batch, model, optimizer, criterion, global_step, updated_step):
    model.train()
    with autocast():
        batch = {k: v.to(args.device) for k, v in batch.items()}

        outputs = model(**batch)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)

        loss = criterion(logits, batch['labels'])
        acc = torch.sum(preds.cpu() == batch['labels'].cpu())

        loss.backward()
        if (global_step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            updated_step += 1

    return loss.item(), acc.item(), updated_step


def evaluating(args, eval_dataloader, model, criterion):
    model.eval()

    # eval phase
    eval_loss = AverageMeter()
    eval_acc = AverageMeter()

    eval_iterator = tqdm(eval_dataloader, desc='eval-Iteration')
    for step, batch in enumerate(eval_iterator):
        batch = {k: v.to(args.device) for k, v in batch.items()}

        outputs = model(**batch)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)

        loss = criterion(logits, batch['labels'])
        acc = torch.sum(preds.cpu() == batch['labels'].cpu())

        eval_loss.update(loss.item(), args.per_device_eval_batch_size)
        eval_acc.update(acc.item() / args.per_device_eval_batch_size)

        description = f'eval loss: {eval_loss.avg:.4f} | eval acc: {eval_acc.avg:.4f}'
        eval_iterator.set_description(description)

    eval_loss = eval_loss.avg
    eval_acc = eval_acc.avg

    return eval_loss, eval_acc


if __name__ == '__main__':
    main()

