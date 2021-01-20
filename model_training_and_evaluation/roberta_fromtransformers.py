# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning multi-lingual models on XNLI (e.g. Bert, DistilBERT, XLM).
    Adapted from `examples/text-classification/run_glue.py`"""


import argparse
import glob
import logging
import os
import random
from math import ceil
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from typing import List
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import os
import pandas

from sklearn.metrics import f1_score
import transformers
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import xnli_compute_metrics as compute_metrics
from transformers import xnli_output_modes as output_modes


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


def load_in_pickled_model(filename, cuda_device, num_labels, label_weights, lowercase_all_text):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config = AutoConfig.from_pretrained(
            'roberta-base',
            num_labels=num_labels,
            finetuning_task="classification",
            cache_dir=None,
        )
        setattr(config, "num_labels", num_labels)

        tokenizer = AutoTokenizer.from_pretrained(
            'roberta-base',
            do_lower_case=lowercase_all_text,
            cache_dir=None,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            filename,
            from_tf=False,
            config=config,
            cache_dir=None,
        )
    warnings.filterwarnings('default')

    if cuda_device != -1 and torch.cuda.is_available():
        if cuda_device == -1:
            model = model.to(torch.device("cuda"))
        else:
            model = model.to(torch.device(f"cuda:{cuda_device}"))
    return model, tokenizer


# from https://stackoverflow.com/questions/800197/how-to-get-all-of-the-immediate-subdirectories-in-python
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def load_in_fname_to_var_key(filename='file_date.csv'):
    csv_file = pandas.read_csv(filename, dtype={'file_name': str, 'start_year': np.float32, 'start_month': np.float32})
    fname_to_var = {}
    for i, row in csv_file.iterrows():
        fname = row['file_name']
        startyear = int(row['start_year'])
        startmonth = int(row['start_month'])
        var = startyear - 1970 + ((startmonth - 1) / 12)
        fname_to_var[fname] = var
    return fname_to_var


def convert_fname_to_continuous_variable(fname_to_var, fname_with_slash):
    fname_parts = fname_with_slash.split('/')
    return fname_to_var[fname_parts[0]]


def get_subdir_with_model_to_load(output_dir):
    if not output_dir.endswith('/'):
        output_dir += '/'

    # since we used early stopping during training, we need to find the name of the directory that represents
    # the last saved model
    checkpoints = get_immediate_subdirectories(output_dir)
    assert len(checkpoints) > 0, 'FOUND NO IMMEDIATE SUBDIRECTORIES OF ' + output_dir
    last_epoch_found = -1
    inds_of_last_epoch = []
    single_hyphen_only = False
    for i, checkpoint in enumerate(checkpoints):
        if single_hyphen_only or checkpoint.count('-') < 3:
            assert checkpoint.count('-') == 1
            single_hyphen_only = True
            continue  # not in the right format to qualify
        epoch = int(checkpoint[checkpoint.rfind('-') + 1:])
        if epoch > last_epoch_found:
            last_epoch_found = epoch
            inds_of_last_epoch = [i]
        elif epoch == last_epoch_found:
            inds_of_last_epoch.append(i)

    if single_hyphen_only:
        biggest_num_so_far = -1
        winning_subdir = None
        for i, checkpoint in enumerate(checkpoints):
            num = int(checkpoint[checkpoint.rfind('-') + 1:])
            if num > biggest_num_so_far:
                biggest_num_so_far = num
                winning_subdir = checkpoint
        output_dir = os.path.join(output_dir, winning_subdir)
        if not output_dir.endswith('/'):
            output_dir += '/'
        return output_dir

    else:
        biggest_earlier_num = -1
        winning_subdir = None
        for ind in inds_of_last_epoch:
            subdir = checkpoints[ind]
            num = subdir[subdir.index('-') + 1:]
            num = int(num[:num.index('-')])
            if num > biggest_earlier_num:
                winning_subdir = subdir

        output_dir = os.path.join(output_dir, winning_subdir)
        if not output_dir.endswith('/'):
            output_dir += '/'
        return output_dir


def run_best_model_on(output_dir, test_df, num_labels, label_weights, lowercase_all_text,
                      cuda_device=-1, string_prefix='', f1_avg: str='weighted', print_results=True,
                      also_report_binary_precrec=False):
    output_dir = get_subdir_with_model_to_load(output_dir)

    if cuda_device == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', cuda_device)

    model, tokenizer = load_in_pickled_model(output_dir, cuda_device, num_labels, label_weights, lowercase_all_text)
    result, model_outputs = evaluate(model, tokenizer, also_report_binary_precrec, "Classifying", output_dir,
                                     cuda_device, 4, 1, device, "roberta", "classification", test_df, prefix="",
                                     f1_avg_type=f1_avg, return_model_outputs_too=True)

    if print_results:
        if also_report_binary_precrec:
            print(string_prefix + 'RoBERTa result: accuracy is ' + str(result['acc']) + ' and ' + f1_avg + ' f1 is ' +
                  str(result['f1']) + ' (precision is ' + str(result['prec']) + ' and recall is ' +
                  str(result['rec']) + ')')
        else:
            print(string_prefix + 'RoBERTa result: accuracy is ' + str(result['acc']) + ' and ' + f1_avg + ' f1 is ' +
                  str(result['f1']))
    if also_report_binary_precrec:
        return result['f1'], result['acc'], list(model_outputs), result['prec'], result['rec']
    else:
        return result['f1'], result['acc'], list(model_outputs)


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def train(train_dataset, model, tokenizer, calc_binary_precrec_too, f1_avg_type, seed, n_gpu, max_steps,
          local_rank, per_gpu_train_batch_size, gradient_accumulation_steps, num_train_epochs, weight_decay,
          learning_rate, adam_epsilon, warmup_steps, model_name_or_path, fp16, fp16_opt_level, device,
          model_type, max_grad_norm, logging_steps, evaluate_during_training, save_steps, per_gpu_eval_batch_size,
          task_name, output_mode, output_dir, dev_df, evaluate_at_end_of_epoch, save_at_end_of_epoch,
          metric_to_improve_on, higher_metric_is_better, class_weights, num_classes, dict_of_info_for_debugging):
    """ Train the model """
    if local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    if isinstance(train_dataset, tuple):
        fname_vars = train_dataset[1]
        train_dataset = train_dataset[0]

    train_batch_size = per_gpu_train_batch_size * max(1, n_gpu)
    train_sampler = RandomSampler(train_dataset)# if local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

    if max_steps > 0:
        t_total = max_steps
        num_train_epochs = max_steps // (len(train_dataloader) // gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(model_name_or_path, "scheduler.pt")))

    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    """if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True
        )"""

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        train_batch_size
        * gradient_accumulation_steps
        * 1, #(torch.distributed.get_world_size() if local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    best_metric_val_seen = None
    just_reached_best_val_so_far = False
    if class_weights is None:
        loss_function = torch.nn.CrossEntropyLoss()
    else:
        print('Class weights: ' + str(class_weights))
        weight_tensor = torch.tensor(class_weights, dtype=torch.float)
        weight_tensor = weight_tensor.to(next(model.parameters()).device)
        loss_function = torch.nn.CrossEntropyLoss(weight=weight_tensor)

    # Check if continuing training from a checkpoint
    if os.path.exists(model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(num_train_epochs), desc="Epoch", disable=local_rank not in [-1, 0]
    )
    set_seed(seed, n_gpu)  # Added here for reproductibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            """if model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if model_type in ["bert"] else None
                )  # XLM and DistilBERT don't use segment_ids"""
            """print("Min val in input_ids is " + str(int(torch.min(inputs['input_ids']))) + ' and max is ' +
                  str(int(torch.max(inputs['input_ids']))) + ', max length of sequence is ' +
                  str(int(torch.max(inputs['attention_mask'].sum(1)))) + ' (embedding matrix size is ' +
                  str(model.roberta.embeddings.word_embeddings.weight.shape[0]) + ' and num of position ids is ' +
                  str(model.roberta.embeddings.position_embeddings.weight.shape[0]) + ')')"""
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            # recalculate loss using class weights here
            logits = outputs[1]
            loss = loss_function(logits.view(-1, num_classes), inputs['labels'].view(-1))

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % gradient_accumulation_steps == 0:
                if fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

            if ((local_rank in [-1, 0] and logging_steps > 0 and global_step % logging_steps == 0) or
                    (evaluate_at_end_of_epoch and step == len(train_dataloader) - 1)):
                # Log metrics
                if (
                        (local_rank == -1 and evaluate_during_training) or
                        (evaluate_at_end_of_epoch and step == len(train_dataloader) - 1)
                ):  # Only evaluate when single GPU otherwise metrics may not average well
                    results = evaluate(model, tokenizer, calc_binary_precrec_too, task_name, output_dir, local_rank,
                                       per_gpu_eval_batch_size, n_gpu, device, model_type, output_mode, dev_df,
                                       f1_avg_type=f1_avg_type)
                    for key, value in results.items():
                        tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    if (best_metric_val_seen is None or
                            (higher_metric_is_better and results[metric_to_improve_on] > best_metric_val_seen) or
                          ((not higher_metric_is_better) and results[metric_to_improve_on] < best_metric_val_seen)):
                        best_metric_val_seen = results[metric_to_improve_on]
                        just_reached_best_val_so_far = True
                tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar("loss", (tr_loss - logging_loss) / logging_steps, global_step)
                logging_loss = tr_loss
                """print('Just logged metrics and just_reached_best_val_so_far: ' + str(just_reached_best_val_so_far) +
                      ', value is ' + str(best_metric_val_seen))"""

            if just_reached_best_val_so_far:
                """((local_rank in [-1, 0] and save_steps > 0 and global_step % save_steps == 0) or
                    (save_at_end_of_epoch and step == len(train_dataloader) - 1) or just_reached_best_val_so_far):"""
                # Save model checkpoint
                just_reached_best_val_so_far = False
                cur_output_dir = os.path.join(output_dir, "checkpoint-{}".format(global_step))
                if not os.path.exists(cur_output_dir):
                    os.makedirs(cur_output_dir)
                """model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training"""
                model.save_pretrained(cur_output_dir)
                tokenizer.save_pretrained(cur_output_dir)

                logger.info("Saving model checkpoint to %s", cur_output_dir)

                torch.save(optimizer.state_dict(), os.path.join(cur_output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(cur_output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", cur_output_dir)

            just_reached_best_val_so_far = False

            if max_steps > 0 and global_step > max_steps:
                epoch_iterator.close()
                break
        if max_steps > 0 and global_step > max_steps:
            train_iterator.close()
            break

    if local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def simple_accuracy(preds: np.array, labels: np.array):
    return (preds == labels).astype(float).mean()


def f1(preds: np.array, labels: np.array, f1_avg="weighted"):
    return f1_score(labels, preds, average=f1_avg)


def calc_precision(preds: np.array, labels: np.array, f1_avg="weighted"):
    return precision_score(labels, preds, average=f1_avg)


def calc_recall(preds: np.array, labels: np.array, f1_avg="weighted"):
    return recall_score(labels, preds, average=f1_avg)


def evaluate(model, tokenizer, calc_binary_precrec_too, task_name, output_dir, local_rank,
             per_gpu_eval_batch_size, n_gpu, device, model_type, output_mode, eval_df, prefix="",
             f1_avg_type="weighted", return_model_outputs_too=False):
    eval_task_names = (task_name,)
    eval_outputs_dirs = (output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(eval_df, tokenizer,
                                               includes_context_column='contextbefore' in eval_df.columns)
        if isinstance(eval_dataset, tuple):
            fname_vars = eval_dataset[1]
            eval_dataset = eval_dataset[0]

        if not os.path.exists(eval_output_dir) and local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        eval_batch_size = per_gpu_eval_batch_size * max(1, n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)

        # multi-gpu eval
        if n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                """if model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if model_type in ["bert"] else None
                    )  # XLM and DistilBERT don't use segment_ids"""
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        raw_model_logits = preds
        if output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        else:
            raise ValueError("No other `output_mode` for XNLI.")
        # preds, out_label_ids
        result = {"acc": simple_accuracy(preds, out_label_ids), "f1": f1(preds, out_label_ids, f1_avg=f1_avg_type)}
        if calc_binary_precrec_too:
            result['prec'] = calc_precision(preds, out_label_ids, f1_avg=f1_avg_type)
            result['rec'] = calc_recall(preds, out_label_ids, f1_avg=f1_avg_type)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if return_model_outputs_too:
        return results, list(raw_model_logits)
    else:
        return results


def convert_list_of_inputs_id_tensors_to_single_tensor(list_of_input_id_tensors, padding_token):
    if len(list_of_input_id_tensors) == 0:
        return torch.zeros((0, 0), dtype=torch.long)
    if isinstance(list_of_input_id_tensors[0], torch.Tensor):
        max_num_tokens = 0
        for input_tensor in list_of_input_id_tensors:
            cur_num_tokens = torch.squeeze(input_tensor).size(0)
            if cur_num_tokens > max_num_tokens:
                max_num_tokens = cur_num_tokens
        tensor_to_return = torch.zeros((len(list_of_input_id_tensors), max_num_tokens), dtype=torch.long) + \
                           int(padding_token)
        for i, input_tensor in enumerate(list_of_input_id_tensors):
            tensor_to_insert = torch.squeeze(input_tensor)
            tensor_to_return[i][0: tensor_to_insert.size(0)] = tensor_to_insert
    else:
        assert isinstance(list_of_input_id_tensors[0], list)
        max_num_tokens = 0
        for input_tensor in list_of_input_id_tensors:
            cur_num_tokens = len(input_tensor)
            if cur_num_tokens > max_num_tokens:
                max_num_tokens = cur_num_tokens
        tensor_to_return = torch.zeros((len(list_of_input_id_tensors), max_num_tokens), dtype=torch.long) + \
                           int(padding_token)
        for i, input_tensor in enumerate(list_of_input_id_tensors):
            tensor_to_insert = torch.tensor(input_tensor, dtype=torch.long)
            tensor_to_return[i][0: tensor_to_insert.size(0)] = tensor_to_insert
    return tensor_to_return


def get_mask_from_sequence_lengths(
    sequence_lengths: torch.Tensor, max_length: int
) -> torch.BoolTensor:
    """
    Given a variable of shape `(batch_size,)` that represents the sequence lengths of each batch
    element, this function returns a `(batch_size, max_length)` mask variable.  For example, if
    our input was `[2, 2, 3]`, with a `max_length` of 4, we'd return
    `[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]`.
    We require `max_length` here instead of just computing it from the input `sequence_lengths`
    because it lets us avoid finding the max, then copying that value from the GPU to the CPU so
    that we can use it to construct a new tensor.
    """
    # (batch_size, max_length)
    ones = sequence_lengths.new_ones((sequence_lengths.size(0), max_length))
    range_tensor = ones.cumsum(dim=1)
    return sequence_lengths.unsqueeze(1) >= range_tensor


def load_and_cache_examples(data_df, tokenizer, includes_context_column=False):
    fname_to_var_key = load_in_fname_to_var_key()
    padding_token = tokenizer.pad_token_type_id
    if padding_token != 1:
        print('Padding token was listed as ' + str(padding_token) + ', but that maps to ' +
              str(tokenizer.decode([padding_token])) + '. Switching padding token to 1 instead, which maps to ' +
              str(tokenizer.decode([1])))
        padding_token = 1
    if includes_context_column:
        # columns: text, [strlabel,] labels, contextbefore
        column_to_fetch = data_df.shape[1] - 1
        input_ids_for_text = []
        num_in_context_before = []
        max_length = 0
        for i, row in data_df.iterrows():
            contextbefore_tokens = tokenizer.encode(str(row['contextbefore']).strip(), add_special_tokens=True)
            num_in_context_before.append(len(contextbefore_tokens))
            fully_encoded_text = contextbefore_tokens + \
                                 (tokenizer.encode(str(row['text']).strip(), add_special_tokens=True)[1:])
            if len(fully_encoded_text) > 512:
                fully_encoded_text = [fully_encoded_text[0]] + fully_encoded_text[-511:]
            if len(fully_encoded_text) > max_length:
                max_length = len(fully_encoded_text)
            input_ids_for_text.append(fully_encoded_text)
        input_ids_for_text = convert_list_of_inputs_id_tensors_to_single_tensor(input_ids_for_text, padding_token)
    else:
        # columns: text, [strlabel,] labels
        input_ids_for_text = []
        for i, row in data_df.iterrows():
            fully_encoded_text = tokenizer.encode(str(row['text']).strip(), add_special_tokens=True)
            if len(fully_encoded_text) > 512:
                fully_encoded_text = [fully_encoded_text[0]] + fully_encoded_text[-511:]
            input_ids_for_text.append(fully_encoded_text)
        input_ids_for_text = convert_list_of_inputs_id_tensors_to_single_tensor(input_ids_for_text, padding_token)

    # Convert to Tensors and build dataset
    all_attention_mask = (input_ids_for_text != padding_token).long()
    all_token_type_ids = torch.zeros(input_ids_for_text.size(), dtype=torch.long)
    if includes_context_column:
        all_token_type_ids = all_token_type_ids + 1
        one_where_context = get_mask_from_sequence_lengths(torch.tensor(num_in_context_before), max_length).long()
        all_token_type_ids = all_token_type_ids - one_where_context
    all_labels = torch.tensor([row['labels'] for i, row in data_df.iterrows()], dtype=torch.long)
    if 'filename' in data_df.columns:
        fname_vars = torch.tensor([convert_fname_to_continuous_variable(fname_to_var_key, row['filename'])
                                   for i, row in data_df.iterrows()], dtype=torch.float)

    dataset = TensorDataset(input_ids_for_text, all_attention_mask, all_token_type_ids, all_labels)
    if 'filename' in data_df.columns:
        return dataset, fname_vars
    else:
        return dataset


def find_latest_metric_in_dict(dict_to_query, metric_name):
    all_keys = list(dict_to_query.keys())
    metric_name_with_underscore_present = False
    for key in all_keys:
        if key.startswith(metric_name + '_'):
            metric_name_with_underscore_present = True
    if metric_name_with_underscore_present:
        # find the latest one of these
        all_vals_after = []
        for key in all_keys:
            if key.startswith(metric_name + '_'):
                part_after = key[len(metric_name + '_'):]
                if part_after.strip() != '':
                    all_vals_after.append(int(part_after))
        all_vals_after = sorted(all_vals_after, reverse=True)
        if len(all_vals_after) > 0:
            return dict_to_query[metric_name + '_' + str(all_vals_after[0])]
        else:
            return dict_to_query[metric_name + '_']
    else:
        if metric_name not in dict_to_query:
            print(dict_to_query)
        return dict_to_query[metric_name]


def run_classification(train_df, dev_df, num_labels, output_dir, batch_size: int, learning_rate: float,
                       label_weights: List[float]=None, string_prefix='',
                       lowercase_all_text=True, cuda_device=-1, f1_avg='weighted', test_set=None,
                       print_results=True, also_report_binary_precrec=False):
    print('torch.cuda.is_available: ' + str(torch.cuda.is_available()))
    print('cuda device: ' + str(cuda_device))
    # label_weights aren't accommodated in huggingface's RobertaForSequenceClassification
    model_name_or_path = "roberta-base"  # Path to pretrained model or model identifier from huggingface.co/models
    language = None  # Evaluation language. Also train language if `train_language` is set to None.
    train_language = None
    config_name = ""  # Pretrained config name or path if not the same as model_name
    tokenizer_name = ""  # Pretrained tokenizer name or path if not the same as model_name
    cache_dir = None  # Where do you want to store the pre-trained models downloaded from huggingface.co
    max_seq_length = 128
    do_train = True
    do_eval = True
    evaluate_during_training = False
    do_lower_case = lowercase_all_text
    per_gpu_train_batch_size = 3  # batch_size
    per_gpu_eval_batch_size = 3  # batch_size
    gradient_accumulation_steps = ceil(batch_size / 8)
    learning_rate = learning_rate # 5e-5
    weight_decay = 0.0
    adam_epsilon = 1e-8
    max_grad_norm = 1.0
    num_train_epochs = 10.0
    max_steps = -1
    warmup_steps = 0
    logging_steps = 500
    save_steps = None  # we don't use this anymore
    eval_all_checkpoints = False
    overwrite_output_dir = False
    overwrite_cache = False
    seed = 42
    fp16 = False
    fp16_opt_level = "01"
    evaluate_at_end_of_epoch = True
    save_at_end_of_epoch = True
    metric_to_perform_better_on = 'f1'
    higher_is_better = True

    if (
            os.path.exists(output_dir)
            and os.listdir(output_dir)
            and do_train
            and not overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                output_dir
            )
        )

    # Setup CUDA, GPU & distributed training
    if cuda_device == -1:
        no_cuda = True
        local_rank = cuda_device
        n_gpu = 0
        device = torch.device("cpu")
    else:
        no_cuda = False
        local_rank = cuda_device
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        #torch.distributed.init_process_group(backend="nccl")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        local_rank,
        device,
        n_gpu,
        bool(local_rank != -1),
        fp16,
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if __name__ == '__main__':
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    # Set seed
    set_seed(seed, n_gpu)

    # Prepare XNLI task
    task_name = "finetuning"
    output_mode = "classification"

    # Load pretrained model and tokenizer
    if local_rank not in [-1, 0]:
        pass
        #torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        cache_dir=cache_dir,
    )
    model_type = config.model_type
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        do_lower_case=do_lower_case,
        cache_dir=cache_dir,
    )

    config_to_initialize_classification_model = config
    model = transformers.RobertaForSequenceClassification(config_to_initialize_classification_model)
    model.roberta = transformers.RobertaModel.from_pretrained(model_name_or_path)

    if local_rank == 0:
        pass
        #torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(device)
    print("model is on cuda: " + str(next(model.parameters()).device))

    # Training
    if do_train:
        dict_of_info_for_debugging = {'cuda_device': cuda_device, 'num_labels': num_labels,
                                      'label_weights': label_weights, 'lowercase_all_tokens': lowercase_all_text}
        train_dataset = load_and_cache_examples(train_df, tokenizer,
                                                includes_context_column='contextbefore' in train_df.columns)
        global_step, tr_loss = train(train_dataset, model, tokenizer, also_report_binary_precrec, f1_avg,
                                     seed, n_gpu, max_steps, local_rank, per_gpu_train_batch_size,
                                     gradient_accumulation_steps, num_train_epochs, weight_decay,
                                     learning_rate, adam_epsilon, warmup_steps, model_name_or_path, fp16,
                                     fp16_opt_level, device, model_type, max_grad_norm, logging_steps,
                                     evaluate_during_training, save_steps, per_gpu_eval_batch_size, task_name,
                                     output_mode, output_dir, dev_df, evaluate_at_end_of_epoch, save_at_end_of_epoch,
                                     metric_to_perform_better_on, higher_is_better, label_weights, num_labels,
                                     dict_of_info_for_debugging)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if do_train and True:#(local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        """model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training"""
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model

        # Load a trained model and vocabulary that you have fine-tuned
        dir_to_load = get_subdir_with_model_to_load(output_dir)
        model, tokenizer = load_in_pickled_model(dir_to_load, cuda_device, num_labels, label_weights, lowercase_all_text)

    # Evaluation
    results = {}
    if do_eval and local_rank in [-1, 0]:
        checkpoints = [output_dir]
        if eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )

        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            dir_to_load = get_subdir_with_model_to_load(checkpoint)
            model, tokenizer = load_in_pickled_model(dir_to_load, cuda_device, num_labels, label_weights,
                                                     lowercase_all_text)
            if test_set is None:
                result = evaluate(model, tokenizer, also_report_binary_precrec, task_name, output_dir, local_rank,
                                  per_gpu_eval_batch_size, n_gpu, device, model_type, output_mode, dev_df,
                                  prefix=prefix, f1_avg_type=f1_avg)
            else:
                result = evaluate(model, tokenizer, also_report_binary_precrec, task_name, output_dir, local_rank,
                                  per_gpu_eval_batch_size, n_gpu, device, model_type, output_mode, test_set,
                                  prefix=prefix, f1_avg_type=f1_avg)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    f1 = find_latest_metric_in_dict(results, 'f1')
    acc = find_latest_metric_in_dict(results, 'acc')
    model_outputs = None
    if also_report_binary_precrec:
        prec = find_latest_metric_in_dict(results, 'prec')
        rec = find_latest_metric_in_dict(results, 'rec')
    if print_results:
        if also_report_binary_precrec:
            print(string_prefix + 'With batch size ' + str(batch_size) + ' and learning rate ' + str(learning_rate) +
                  ', RoBERTa result: accuracy is ' + str(acc) + ' and ' + f1_avg + ' f1 is ' +
                  str(f1) + ' (precision is ' + str(prec) + ' and recall is ' +
                  str(rec) + ')')
        else:
            print(string_prefix + 'With batch size ' + str(batch_size) + ' and learning rate ' + str(learning_rate) +
                  ', RoBERTa result: accuracy is ' + str(acc) + ' and ' + f1_avg + ' f1 is ' +
                  str(f1))
    if also_report_binary_precrec:
        return f1, acc, model_outputs, prec, rec
    else:
        return f1, acc, model_outputs


if __name__ == '__main__':
    load_in_fname_to_var_key()