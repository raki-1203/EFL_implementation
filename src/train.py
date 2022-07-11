import argparse
import json
import logging
import math
import os
import sys
import random
import logging

import wandb
import numpy as np
import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score

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

from src.utils import save_pickle, load_pickle


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
        default=5e-5,
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
    data_files = {}
    if args.train_file is not None:
        data_files['train'] = os.path.join(data_dir, args.train_file)
    if args.validation_file is not None:
        data_files['validation'] = os.path.join(data_dir, args.validation_file)
    extension = (args.train_file if args.train_file is not None else args.validation_file).split('.')[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)

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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Preprocessing the datasets
    non_label_column_names = [name for name in raw_datasets['train'].column_names if name != 'label']
    if 'sentence1' in non_label_column_names and 'sentence2' in non_label_column_names:
        sentence1_key, sentence2_key = 'sentence1', 'sentence2'
    else:
        if len(non_label_column_names) >= 2:
            sentence1_key, sentence2_key = non_label_column_names[:2]
        else:
            sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = {v: i for i, v in enumerate(label_list)}

    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = 'max_length' if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key], ) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
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

    if not os.path.isfile(os.path.join(data_dir, 'processed_datasets.pkl')):  # 7분 걸려서 미리 해두자
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets['train'].column_names,
            desc='Running tokenizer on dataset',
        )
        save_pickle(os.path.join(data_dir, 'processed_datasets.pkl'), processed_datasets)
    else:
        processed_datasets = load_pickle(os.path.join(data_dir, 'processed_datasets.pkl'))

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets['train'].column_names,
        desc='Running tokenizer on dataset',
    )

    train_dataset = processed_datasets['train']
    eval_dataset = processed_datasets['validation']

    # Log a few random samples from the training set:
    # for index in random.sample(range(len(train_dataset)), 3):
    #     logger.info(f'Sample {index} of the training set: {train_dataset[index]}.')

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

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

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

    # Get the metric function
    metric = load_metric('accuracy')

    # Train!
    total_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps

    logger.info('***** Running training *****')
    logger.info(f'  Num examples = {len(train_dataset)}')
    logger.info(f'  Num Epochs = {args.num_train_epochs}')
    logger.info(f'  Instantaneous batch size per device = {args.per_device_train_batch_size}')
    logger.info(f'  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}')
    logger.info(f'  Gradient Accumulation steps = {args.gradient_accumulation_steps}')
    logger.info(f'  Total optimization steps = {args.max_train_steps}')

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(int(args.max_train_steps)))
    completed_steps = 0
    starting_epoch = 0

    if isinstance(checkpointing_steps, int):
        best_checkpoining_steps = None
        best_valid_loss = float('inf')

    for epoch in range(starting_epoch, args.num_train_epochs):
        print(f'\n{epoch+1} epoch start!')
        model.train()
        if args.with_tracking:
            total_loss = 0
            total_correct = 0
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.item()
                predictions = outputs.logits.detach().cpu().argmax(dim=-1)
                total_correct += torch.sum(batch['labels'].cpu() == predictions)
                cur_step_acc = total_correct / ((step + 1) * args.per_device_train_batch_size) * 100
                progress_bar.set_description(f'loss: {total_loss / (step + 1):.4f} | accuracy: {cur_step_acc:.4f}%')
                wandb.log({'train_loss': total_loss / (step + 1), 'train_acc': cur_step_acc})
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    model.eval()
                    eval_total_loss = 0
                    eval_total_correct = 0
                    for step, batch in enumerate(tqdm(eval_dataloader, desc='evaluate')):
                        batch = {k: v.to(device) for k, v in batch.items()}

                        with torch.no_grad():
                            outputs = model(**batch)

                        loss = outputs.loss
                        eval_total_loss += loss.item()
                        predictions = outputs.logits.argmax(dim=-1).cpu()
                        eval_total_correct += torch.sum(batch['labels'].cpu() == predictions)

                    if args.with_tracking:
                        train_acc = total_correct / (len(train_dataloader) * args.per_device_train_batch_size)
                        train_loss = total_loss / len(train_dataloader)
                        valid_acc = eval_total_correct / (len(eval_dataloader) * args.per_device_eval_batch_size) * 100
                        valid_loss = eval_total_loss / len(eval_dataloader)
                        logger.info(f'epoch {epoch} | eval_accuracy: {valid_acc:.4f} | eval_loss: {valid_loss:.6f}')
                        wandb.log({
                            'train_acc': train_acc,
                            'train_loss': train_loss,
                            'valid_acc': valid_acc,
                            'valid_loss': valid_loss,
                        })

                    if best_valid_loss > valid_loss:
                        best_valid_loss = valid_loss
                        best_checkpoining_steps = checkpointing_steps
                        output_dir = f'step_{completed_steps}'
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        model.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        # if epoch < args.num_train_epochs - 1:
        #     model.save_pretrained(args.output_dir)
        #     tokenizer.save_pretrained(args.output_dir)
        #
        # if args.checkpointing_steps == 'epoch':
        #     output_dir = f'epoch_{epoch}'
        #     if args.output_dir is not None:
        #         output_dir = os.path.join(args.output_dir, output_dir)
        #     model.save_pretrained(output_dir)

    if args.output_dir is not None:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    # load model at best checkpoint step
    model.from_pretrained(os.path.join(args.output_dir, f'step_{best_checkpoining_steps}'))

    # Final evaluation on kornli validation set
    model.eval()
    eval_total_correct = 0
    for step, batch in enumerate(eval_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)

        predictions = outputs.logits.argmax(dim=-1).cpu()
        eval_total_correct += torch.sum(batch['labels'].cpu() == predictions)

    valid_acc = eval_total_correct / (len(eval_dataloader) * args.per_device_eval_batch_size) * 100
    logger.info(f'\nxnli devset: accuracy: {valid_acc:.4f}')

    if args.output_dir is not None:
        with open(os.path.join(args.output_dir, 'all_results.json'), 'w') as f:
            json.dump({'eval_accuracy': valid_acc}, f)

    if args.with_tracking:
        wandb.finish()


if __name__ == '__main__':
    main()

