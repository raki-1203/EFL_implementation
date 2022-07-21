import os
import sys
import logging
import json
import math
import wandb
import numpy as np

import torch

from datasets import load_from_disk
from torch import nn
from torch.cuda.amp import autocast
from tqdm.auto import tqdm


project_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(project_dir, 'data')

sys.path.append(project_dir)

from src.utils import (
    AverageMeter,
    make_dataset,
    get_config_tokenizer_model,
    get_optimizer_grouped_parameters,
    get_lr_scheduler,
    set_seed,
    get_dataloader,
)
from src.data import processor_dict
from src.loss import RDropLoss
from src.task_label_description import TASK_LABELS_DESC
from src.arguments import parse_args

# wandb description silent
os.environ['WANDB_SILENT'] = "true"

# gpu setting
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Set the GPU 0 to use

logger = logging.getLogger(__name__)


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
    if not os.path.exists(os.path.join(data_dir, args.task_dataset + '_dataset')):
        make_dataset(args)
    raw_datasets = load_from_disk(os.path.join(data_dir, args.task_dataset + '_dataset'))

    # Labels
    if args.task_dataset == 'kornli':
        # A useful fast method
        label_list = raw_datasets['train'].unique('label')
        label_list.sort()  # Let's sort it for determinism
        num_labels = len(label_list)
    else:
        label_list = ['not entail', 'entail']
        num_labels = len(label_list)

    # Load pretrained model and tokenizer
    config, tokenizer, model = get_config_tokenizer_model(args, num_labels)
    model.to(args.device)

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_idx = {v: i for i, v in enumerate(label_list)}

    model.config.label2id = label_to_idx
    model.config.id2label = {idx: label for label, idx in config.label2id.items()}

    processor = processor_dict[args.task_dataset](args.negative_num)
    processed_dataset = processor.create_examples(raw_datasets,
                                                  TASK_LABELS_DESC[args.task_dataset])

    padding = 'max_length' if args.pad_to_max_length else False

    if len(raw_datasets.keys()) == 2:
        train_dataloader, eval_dataloader = get_dataloader(args, processed_dataset, tokenizer, padding)
        test_dataloader = None
    else:
        train_dataloader, eval_dataloader, test_dataloader = get_dataloader(args, processed_dataset, tokenizer, padding)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(args, model)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

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

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    rdrop_loss = RDropLoss()
    lr_scheduler = get_lr_scheduler(args, optimizer)

    # We need to initialize the trackers we use, and also store our configuration.
    if args.with_tracking:
        experiment_config = vars(args)
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

    global_step = 0
    starting_epoch = 0
    train_loss = AverageMeter()
    train_acc = AverageMeter()

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None and "step" in args.resume_from_checkpoint or \
                "epoch" in args.resume_from_checkpoint:
            logger.info(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            model.load_state_dict(torch.load(args.resume_from_checkpoint))
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [os.path.join(args.resume_from_checkpoint, f.name) for f in os.scandir(args.output_dir) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0].split('/')[-1]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
        global_step += starting_epoch * len(train_dataloader)

    if isinstance(checkpointing_steps, int):
        best_checkpoining_steps = None
        best_valid_acc = 0

    for epoch in range(starting_epoch, args.num_train_epochs):
        print(f'\n{epoch + 1} epoch start!')
        torch.cuda.empty_cache()

        train_iterator = tqdm(train_dataloader, desc='train-Iteration')
        for step, batch in enumerate(train_iterator):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    global_step += 1
                    continue

            loss, acc = training_per_step(args, batch, model, optimizer, criterion, rdrop_loss, lr_scheduler,
                                          global_step)
            train_loss.update(loss, args.per_device_train_batch_size)
            train_acc.update(acc / args.per_device_train_batch_size)
            global_step += 1
            description = f'{epoch + 1}epoch {int(global_step // args.gradient_accumulation_steps):>5d} / {int(args.max_train_steps):>6d}step | loss: {train_loss.avg:.4f} | acc: {train_acc.avg:.4f} | best_acc: {best_valid_acc:.4f}'
            train_iterator.set_description(description)

            if args.with_tracking:
                last_lr = lr_scheduler.get_last_lr()[0]
                wandb.log({
                    'train/loss': train_loss.avg,
                    'train/acc': train_acc.avg,
                    'train/learning_rate': last_lr,
                })

            if isinstance(checkpointing_steps, int):
                if global_step != 0 and (global_step / args.gradient_accumulation_steps) % checkpointing_steps == 0:
                    valid_acc = evaluating(args, eval_dataloader, model, TASK_LABELS_DESC[args.task_dataset])

                    if args.lr_scheduler_type == 'ReduceLROnPlateau':
                        lr_scheduler.step(valid_acc)

                    if valid_acc > best_valid_acc:
                        best_valid_acc = valid_acc
                        best_checkpoining_steps = global_step
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
                            'eval/best_acc': best_valid_acc,
                            'eval/acc': valid_acc,
                        })
                        train_loss.reset()
                        train_acc.reset()

            if (global_step // args.gradient_accumulation_steps) >= args.max_train_steps:
                break

    if test_dataloader is not None:
        # load model at best checkpoint step
        model = model.from_pretrained(os.path.join(args.output_dir, f'step_{best_checkpoining_steps}'))
        model.to(args.device)

        # Final evaluation on kornli test set
        with torch.no_grad():
            test_acc = evaluating(args, test_dataloader, model, TASK_LABELS_DESC[args.task_dataset])

        logger.info(f'\n{args.task_dataset} test set: accuracy: {test_acc:.4f}')

        if args.output_dir is not None:
            with open(os.path.join(args.output_dir, 'all_results.json'), 'w') as f:
                json.dump({'best_step': best_checkpoining_steps, 'test_accuracy': test_acc}, f)
    else:
        if args.output_dir is not None:
            with open(os.path.join(args.output_dir, 'all_results.json'), 'w') as f:
                json.dump({'best_step': best_checkpoining_steps, 'test_accuracy': best_valid_acc}, f)

    if args.with_tracking:
        wandb.finish()


def training_per_step(args, batch, model, optimizer, criterion, rdrop_loss, lr_scheduler, global_step):
    model.train()
    with autocast():
        batch = {k: v.to(args.device) for k, v in batch.items()}

        outputs = model(**batch)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)

        if args.rdrop_coef > 0:
            logits_2 = model(**batch).logits
            ce_loss = (criterion(logits, batch['labels']) + criterion(logits_2, batch['labels'])) * 0.5
            kl_loss = rdrop_loss(logits, logits_2)
            loss = ce_loss + kl_loss * args.rdrop_coef
        else:
            loss = criterion(logits, batch['labels'])

        acc = torch.sum(preds.cpu() == batch['labels'].cpu())

        loss.backward()
        if (global_step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            if args.lr_scheduler_type != 'ReduceLROnPlateau':
                lr_scheduler.step()

    return loss.item(), acc.item()


def evaluating(args, eval_dataloader, model, task_label_description):
    model.eval()

    # eval phase
    class_num = len(task_label_description)

    # [total_num * class_num, 2]
    all_prediction_probs = []
    # [total_num * class_num]
    all_labels = []

    with torch.no_grad():
        eval_iterator = tqdm(eval_dataloader, desc='eval-Iteration')
        for step, batch in enumerate(eval_iterator):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            input_ids, attention_mask = batch['input_ids'], batch['attention_mask']

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)

            logits = outputs.logits

            all_prediction_probs.append(logits.detach().cpu().numpy())
            all_labels.append(batch['labels'].detach().cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_prediction_probs = np.concatenate(all_prediction_probs, axis=0)
    all_prediction_probs = np.reshape(all_prediction_probs, (-1, class_num, 2))

    prediction_pos_probs = all_prediction_probs[:, :, 1]
    prediction_pos_probs = np.reshape(prediction_pos_probs, (-1, class_num))
    y_pred_index = np.argmax(prediction_pos_probs, axis=-1)

    y_true_index = np.array([true_label_index for idx, true_label_index in enumerate(all_labels)
                             if idx % class_num == 0])

    total_num = len(y_true_index)
    correct_num = (y_pred_index == y_true_index).sum()

    return correct_num / total_num


if __name__ == '__main__':
    main()
