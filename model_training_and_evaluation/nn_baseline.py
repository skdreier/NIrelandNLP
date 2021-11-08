import os
from math import ceil
import shutil
from typing import List
from copy import deepcopy
from util import make_directories_as_necessary
from allennlp_pieces import LSTMSeq2Vec
import logging
from roberta_fromtransformers import get_subdir_with_model_to_load, set_seed, simple_accuracy, calc_recall, \
    calc_precision, f1, load_in_fname_to_var_key, convert_list_of_inputs_id_tensors_to_single_tensor, \
    convert_fname_to_continuous_variable, get_mask_from_sequence_lengths, find_latest_metric_in_dict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm, trange
import torch
import re
import glob
import numpy as np
from string import punctuation, whitespace
if False:
    from gensim.models import Word2Vec
import datetime
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup,
)
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


class FeedForwardLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, p=0.3):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.activation = torch.nn.functional.tanh
        self.dropout = torch.nn.Dropout(p=p)

    def forward(self, tensor):
        return self.dropout(self.activation(self.linear(tensor)))


class NNModel(torch.nn.Module):
    def __init__(self, num_output_labels, pretrained_embeddings=None, label_weights=None,
                 process_context_separately=True,
                 num_hidden_layers=3, dim_of_hidden_layers: List[int] = (200, 100, 50)):
        super().__init__()
        assert num_hidden_layers == len(dim_of_hidden_layers)
        if isinstance(pretrained_embeddings, tuple) and len(pretrained_embeddings) == 2:
            num_embedding_rows = pretrained_embeddings[0]
            num_embedding_cols = pretrained_embeddings[1]
        else:
            num_embedding_rows = pretrained_embeddings.shape[0]
            num_embedding_cols = pretrained_embeddings.shape[1]
        self.embeddings = torch.nn.Embedding(num_embeddings=num_embedding_rows,
                                             embedding_dim=num_embedding_cols, padding_idx=0)
        if (not isinstance(pretrained_embeddings, tuple)) or len(pretrained_embeddings) != 2:
            self.embeddings.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

        self.process_context_separately = process_context_separately

        dim_of_hidden_layers = list(dim_of_hidden_layers)
        if process_context_separately:
            dim_of_hidden_layers.insert(0, num_embedding_cols * 2)
            if len(dim_of_hidden_layers) > 1:
                del dim_of_hidden_layers[1]
            dim_of_hidden_layers = [num_embedding_cols * 2] + dim_of_hidden_layers
        else:
            dim_of_hidden_layers = [num_embedding_cols] + dim_of_hidden_layers
        self.hidden_layers = torch.nn.ModuleList([FeedForwardLayer(dim_of_hidden_layers[i], dim_of_hidden_layers[i + 1])
                                                  for i in range(num_hidden_layers)])
        self.projection_layer = torch.nn.Linear(dim_of_hidden_layers[-1], num_output_labels)

        if label_weights is None:
            self.loss = torch.nn.CrossEntropyLoss()
        else:
            self.loss = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(label_weights))

    def save_pretrained(self, output_dir):
        torch.save(self.state_dict(), os.path.join(output_dir, 'model_params.pkl'))

    def mask_embedded_vectors(self, embedded_text, len_each_instance):
        mask = get_mask_from_sequence_lengths(len_each_instance, embedded_text.size(1)).float()
        mask = mask.to(device=embedded_text.device)
        embedded_text = embedded_text * mask.unsqueeze(2).expand(embedded_text.size())
        return embedded_text, mask

    def forward(self, input_ids, context_ids, len_each_maintext, one_where_context, labels):
        """
        :param input_ids: batchsize x max_len_of_maintext
        :param context_ids: batchsize x max_len_of_contexttext_in_batch
        :param len_each_maintext: one-dimensional LongTensor giving actual (nonpad) len of each maintext in batch
        :param one_where_context: batchsize x (max_maintext_len + max_contextlen) with # of 1s at beginning of each
                                  row equal to the number of context tokens in that instance
        :param labels: one-dimensional LongTensor giving label
        :return:
        """
        num_context_tokens = one_where_context.sum(1)
        if num_context_tokens.sum() == 0:
            using_context = False or self.process_context_separately
            if num_context_tokens.numel() > 1:
                pass
                #print('Have entire context of 0s')
                #assert not self.process_context_separately
        else:
            #print('NOT 0s')
            using_context = True

        embedded_inputs = self.embeddings(input_ids)
        embedded_inputs, mask = self.mask_embedded_vectors(embedded_inputs, len_each_maintext)
        embedded_inputs = embedded_inputs.sum(1)
        assert len(embedded_inputs.size()) == 2, str(embedded_inputs.size())

        # now divide each instance vector by the number of tokens in that instance
        divide_by = mask.sum(1).clamp(min=1e-10).unsqueeze(1)
        assert len(divide_by.size()) == 2, str(divide_by.size())
        divide_by = divide_by.expand(embedded_inputs.size())
        embedded_inputs = embedded_inputs / divide_by
        if using_context:
            embedded_context = self.embeddings(context_ids)
            embedded_context, mask = self.mask_embedded_vectors(embedded_context, num_context_tokens)
            embedded_context = embedded_context.sum(1)
            # now divide each instance vector by the number of tokens in that instance
            divide_by_context = mask.sum(1).clamp(min=1e-10).unsqueeze(1)
            assert len(divide_by_context.size()) == 2, str(divide_by_context.size())
            divide_by_context = divide_by_context.expand(embedded_context.size())
            embedded_context = embedded_context / divide_by_context

            if not self.process_context_separately:
                # now turn each vector in embedded_inputs into a weighted sum (by # tokens) of its vector and its
                # context's vector
                denom = divide_by + divide_by_context
                mult_for_text = divide_by / denom
                mult_for_context = divide_by_context / denom
                embedded_inputs = (mult_for_text * embedded_inputs) + (mult_for_context * embedded_context)
            else:
                embedded_inputs = torch.cat([embedded_inputs, embedded_context], 1)

        for hidden_layer in self.hidden_layers:
            embedded_inputs = hidden_layer(embedded_inputs)
        logits = self.projection_layer(embedded_inputs)

        if labels is not None:
            loss = self.loss(logits, labels)
            return loss, logits
        else:
            return (logits,)


class LSTMNNModel(torch.nn.Module):
    def __init__(self, num_output_labels, pretrained_embeddings=None, label_weights=None,
                 process_context_separately=True,
                 num_hidden_layers=3, dim_of_hidden_layers: List[int] = (200, 100, 50)):
        super().__init__()
        if isinstance(pretrained_embeddings, tuple) and len(pretrained_embeddings) == 2:
            num_embedding_rows = pretrained_embeddings[0]
            num_embedding_cols = pretrained_embeddings[1]
        else:
            num_embedding_rows = pretrained_embeddings.shape[0]
            num_embedding_cols = pretrained_embeddings.shape[1]
        self.embeddings = torch.nn.Embedding(num_embeddings=num_embedding_rows + 1,  # for special separator "token"
                                             embedding_dim=num_embedding_cols, padding_idx=0)
        if (not isinstance(pretrained_embeddings, tuple)) or len(pretrained_embeddings) != 2:
            self.embeddings.weight.data.copy_(torch.from_numpy(
                np.concatenate([pretrained_embeddings, np.random.random((1, num_embedding_cols))], axis=0)))
        self.sep_token_ind = num_embedding_rows

        half_hidden_size = dim_of_hidden_layers[0] // 2
        self.lstm = LSTMSeq2Vec(input_size=num_embedding_cols, half_hidden_size=half_hidden_size, num_layers=2)

        self.projection_layer = torch.nn.Linear(200, num_output_labels)

        if label_weights is None:
            self.loss = torch.nn.CrossEntropyLoss()
        else:
            self.loss = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(label_weights))

    def save_pretrained(self, output_dir):
        torch.save(self.state_dict(), os.path.join(output_dir, 'model_params.pkl'))

    def forward(self, input_ids, context_ids, len_each_maintext, one_where_context, labels):
        """
        :param input_ids: batchsize x max_len_of_maintext
        :param context_ids: batchsize x max_len_of_contexttext_in_batch
        :param len_each_maintext: one-dimensional LongTensor giving actual (nonpad) len of each maintext in batch
        :param one_where_context: batchsize x (max_maintext_len + max_contextlen) with # of 1s at beginning of each
                                  row equal to the number of context tokens in that instance
        :param labels: one-dimensional LongTensor giving label
        :return:
        """
        num_context_tokens = one_where_context.sum(1)
        no_context_tokens = num_context_tokens.sum() == 0
        if no_context_tokens:
            using_context = False
            if num_context_tokens.numel() > 1:
                pass
                #print('Have entire context of 0s')
                #assert not self.process_context_separately
        else:
            #print('NOT 0s')
            using_context = True

        # regardless of whether we're using context or not, we get our inputs into a matrix where each row has the
        # following form:
        # sequence of any nonpad context tokens for row [sep] sequence of any input tokens for row
        new_len_each_sequence = num_context_tokens + len_each_maintext + 1
        max_len_of_new_row = torch.max(new_len_each_sequence)

        if no_context_tokens:
            context_ids = torch.cat([torch.full((input_ids.size(0), 1), self.sep_token_ind, dtype=input_ids.dtype,
                                                device=input_ids.device), input_ids], dim=1)
            input_ids = context_ids
        else:
            context_ids = torch.cat([context_ids[:, :torch.max(num_context_tokens)], torch.zeros((input_ids.size(0),
                                                               max_len_of_new_row - torch.max(num_context_tokens)),
                                                               dtype=context_ids.dtype,
                                                               device=context_ids.device)], dim=1)
            for i in range(input_ids.size(0)):
                context_ids[i, num_context_tokens[i]] = self.sep_token_ind
                context_ids[i, num_context_tokens[i] + 1: num_context_tokens[i] + 1 + len_each_maintext[i]] = \
                    input_ids[i, :len_each_maintext[i]]
            input_ids = context_ids

        embedded_inputs = self.embeddings(input_ids)
        mask = get_mask_from_sequence_lengths(new_len_each_sequence, embedded_inputs.size(1)).float()
        mask = mask.to(device=embedded_inputs.device)

        lstm_outputs = self.lstm(embedded_inputs, mask=mask)
        logits = self.projection_layer(lstm_outputs)

        if labels is not None:
            loss = self.loss(logits, labels)
            return loss, logits
        else:
            return (logits,)


def load_in_pickled_model(dir_to_load, dir_with_vocab_key, cuda_device, num_labels, label_weights, lowercase_all_text,
                          embedding_dim, process_context_separately, use_lstm):
    num_embeddings = 1  # padding
    words_to_inds = {}
    with open(os.path.join(dir_with_vocab_key, 'vocab_in_index_order.txt'), 'r') as f:
        last_thing_added = None
        for line in f:
            token = line.strip()
            words_to_inds[token] = num_embeddings
            last_thing_added = token
            num_embeddings += 1  # includes unk
        del words_to_inds[last_thing_added]  # don't put unk in words_to_inds

    if use_lstm:
        model_to_return = LSTMNNModel(num_output_labels=num_labels,
                                      # since num_embeddings auto-adds 1, we don't do that ourselves here
                                      pretrained_embeddings=(num_embeddings, embedding_dim),
                                      label_weights=label_weights,
                                      process_context_separately=process_context_separately)
    else:
        model_to_return = NNModel(num_output_labels=num_labels,
                                  pretrained_embeddings=(num_embeddings, embedding_dim),
                                  label_weights=label_weights, process_context_separately=process_context_separately)
    model_to_return.load_state_dict(torch.load(os.path.join(dir_to_load, 'model_params.pkl')))

    if cuda_device != -1 and torch.cuda.is_available():
        if cuda_device == -1:
            model_to_return = model_to_return.to(torch.device("cuda"))
        else:
            if not str(cuda_device).startswith('c'):
                model_to_return = model_to_return.to(torch.device(f"cuda:{cuda_device}"))
            else:
                model_to_return = model_to_return.to(torch.device(cuda_device))
    return model_to_return, words_to_inds


def evaluate(model, calc_binary_precrec_too, task_name, output_dir, local_rank,
             per_gpu_eval_batch_size, n_gpu, device, output_mode, eval_df, words_to_inds,
             lowercase_all_text, use_context, prefix="",
             f1_avg_type="weighted", return_model_outputs_too=False, regular_exp=None):
    eval_task_names = (task_name,)
    eval_outputs_dirs = (output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(eval_df, lowercase_all_text=lowercase_all_text,
                                               words_to_inds=words_to_inds,
                                               includes_context_column=use_context,
                                               regular_exp=regular_exp)
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
                inputs = {"input_ids": batch[0], "context_ids": batch[1], "len_each_maintext": batch[2],
                          "one_where_context": batch[3], "labels": batch[4]}

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


def check_vocab_dicts_are_same(original_dict, loaded_dict):
    all_values_in_original_dict = sorted(list(original_dict.items()), key=lambda x: x[1])
    for original_tuple in all_values_in_original_dict:
        word = original_tuple[0]
        orig_index = original_tuple[1]
        if word not in loaded_dict:
            print('"' + word + '" (original index ' + str(orig_index) + ') not in loaded dict')
            exit(1)
        elif loaded_dict[word] != orig_index:
            print('"' + word + '" originally has index ' + str(orig_index) + ', but when reloaded, has index ' +
                  str(loaded_dict[word]))
            exit(1)
    for reloaded_word in loaded_dict.keys():
        if reloaded_word not in original_dict:
            print('"' + reloaded_word + '" in reloaded dict but not in original')
            exit(1)


def train(train_dataset, model, calc_binary_precrec_too, f1_avg_type, seed, n_gpu, max_steps,
          local_rank, per_gpu_train_batch_size, gradient_accumulation_steps, num_train_epochs, weight_decay,
          learning_rate, adam_epsilon, warmup_steps, fp16, fp16_opt_level, device,
          max_grad_norm, logging_steps, evaluate_during_training, save_steps, per_gpu_eval_batch_size,
          task_name, output_mode, output_dir, dev_df, evaluate_at_end_of_epoch, save_at_end_of_epoch,
          metric_to_improve_on, higher_metric_is_better, class_weights, num_classes, dict_of_info_for_debugging,
          words_to_inds, lowercase_all_text, use_context, stop_if_no_improvement_after_x_epochs, use_lstm,
          num_labels,
          regular_exp=None):
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

    """optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]"""
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

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

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(num_train_epochs), desc="Epoch", disable=local_rank not in [-1, 0]
    )
    row_to_check = None
    set_seed(seed, n_gpu)  # Added here for reproductibility
    num_evaluations_since_last_improvement = 0
    early_stop = False
    for _ in train_iterator:
        if early_stop:
            break
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(device) for t in batch)

            inputs = {"input_ids": batch[0], "context_ids": batch[1], "len_each_maintext": batch[2],
                      "one_where_context": batch[3], "labels": batch[4]}

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
            #loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            """if row_to_check is None:
                row_to_check = batch[0][0, 0]
                word_embedding = deepcopy(model.embeddings.weight.data[row_to_check])"""

            # recalculate loss using class weights here
            logits = outputs[1]
            loss = outputs[0]

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

                #projection_weights = deepcopy(model.projection_layer.weight.data[0])
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                #assert torch.any(projection_weights != model.projection_layer.weight.data[0])

                """if row_to_check is not None:
                    word_embedding2 = model.embeddings.weight.data[row_to_check]
                    assert torch.any(word_embedding != word_embedding2), \
                        "We haven't actually adjusted params (checked row vector " + str(row_to_check) + ')'
                    row_to_check = None"""

            if ((local_rank in [-1, 0] and logging_steps > 0 and global_step % logging_steps == 0) or
                    (evaluate_at_end_of_epoch and step == len(train_dataloader) - 1)):
                # Log metrics
                if (
                        (local_rank == -1 and evaluate_during_training) or
                        (evaluate_at_end_of_epoch and step == len(train_dataloader) - 1)
                ):  # Only evaluate when single GPU otherwise metrics may not average well
                    results = evaluate(model, calc_binary_precrec_too, task_name, output_dir, local_rank,
                                       per_gpu_eval_batch_size, n_gpu, device, output_mode, dev_df,
                                       lowercase_all_text=lowercase_all_text,
                                       f1_avg_type=f1_avg_type, regular_exp=regular_exp, words_to_inds=words_to_inds,
                                       use_context=use_context)
                    for key, value in results.items():
                        tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    if (best_metric_val_seen is None or
                            (higher_metric_is_better and results[metric_to_improve_on] > best_metric_val_seen) or
                          ((not higher_metric_is_better) and results[metric_to_improve_on] < best_metric_val_seen)):
                        best_metric_val_seen = results[metric_to_improve_on]
                        just_reached_best_val_so_far = True
                        num_evaluations_since_last_improvement = 0
                    else:
                        num_evaluations_since_last_improvement += 1
                        if num_evaluations_since_last_improvement >= stop_if_no_improvement_after_x_epochs:
                            early_stop = True

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

                logger.info("Saving model checkpoint to %s", cur_output_dir)

                torch.save(optimizer.state_dict(), os.path.join(cur_output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(cur_output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", cur_output_dir)

                """model_for_testing, words_to_inds_for_testing = \
                    load_in_pickled_model(cur_output_dir, output_dir, device, num_labels, class_weights,
                                      lowercase_all_text,
                                      model.embeddings.weight.data.shape[1], use_context, use_lstm=use_lstm)

                results = evaluate(model_for_testing, calc_binary_precrec_too, task_name, output_dir, local_rank,
                                   per_gpu_eval_batch_size, n_gpu, device, output_mode, dev_df,
                                   lowercase_all_text=lowercase_all_text,
                                   f1_avg_type=f1_avg_type, regular_exp=regular_exp, words_to_inds=words_to_inds,
                                   use_context=use_context)
                for key, value in results.items():
                    tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                print('should be identical to results just above this')
                check_vocab_dicts_are_same(words_to_inds, words_to_inds_for_testing)
                print()"""

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


def load_and_cache_examples(data_df, lowercase_all_text, words_to_inds, includes_context_column=None, padding_token=0,
                            regular_exp=None):
    assert includes_context_column is not None
    fname_to_var_key = load_in_fname_to_var_key()
    unk_ind = len(words_to_inds) + 1
    length_of_each_maintext = []
    if includes_context_column:
        # columns: text, [strlabel,] labels, contextbefore
        column_to_fetch = data_df.shape[1] - 1
        input_ids_for_context = []
        input_ids_for_text = []
        num_in_context_before = []
        max_length = 0
        for i, row in data_df.iterrows():
            if lowercase_all_text:
                contextbefore_tokens = tokenize_sentence(str(row['contextbefore']).strip().lower(),
                                                         regular_exp=regular_exp)
            else:
                contextbefore_tokens = tokenize_sentence(str(row['contextbefore']).strip(), regular_exp=regular_exp)
            contextbefore_tokens = [unk_ind if word not in words_to_inds else words_to_inds[word]
                                    for word in contextbefore_tokens]
            num_in_context_before.append(len(contextbefore_tokens))
            input_ids_for_context.append(contextbefore_tokens)

            if lowercase_all_text:
                fully_encoded_text = tokenize_sentence(str(row['text']).strip().lower(), regular_exp=regular_exp)
            else:
                fully_encoded_text = tokenize_sentence(str(row['text']).strip(), regular_exp=regular_exp)
            fully_encoded_text = [unk_ind if word not in words_to_inds else words_to_inds[word]
                                  for word in fully_encoded_text]
            length_of_each_maintext.append(len(fully_encoded_text))
            if len(fully_encoded_text) + len(contextbefore_tokens) > max_length:
                max_length = len(fully_encoded_text)
            input_ids_for_text.append(fully_encoded_text)
        input_ids_for_text = convert_list_of_inputs_id_tensors_to_single_tensor(input_ids_for_text, padding_token)
        input_ids_for_context = convert_list_of_inputs_id_tensors_to_single_tensor(input_ids_for_context, padding_token)
    else:
        # columns: text, [strlabel,] labels
        input_ids_for_text = []
        for i, row in data_df.iterrows():
            if lowercase_all_text:
                fully_encoded_text = tokenize_sentence(str(row['text']).strip().lower(), regular_exp=regular_exp)
            else:
                fully_encoded_text = tokenize_sentence(str(row['text']).strip().lower(), regular_exp=regular_exp)
            fully_encoded_text = [unk_ind if word not in words_to_inds else words_to_inds[word]
                                  for word in fully_encoded_text]
            length_of_each_maintext.append(len(fully_encoded_text))
            input_ids_for_text.append(fully_encoded_text)
        input_ids_for_text = convert_list_of_inputs_id_tensors_to_single_tensor(input_ids_for_text, padding_token)
        input_ids_for_context = torch.zeros((input_ids_for_text.size(0), 1)).long()

    # Convert to Tensors and build dataset
    if includes_context_column:
        one_where_context = get_mask_from_sequence_lengths(torch.tensor(num_in_context_before), max_length).long()
    else:
        one_where_context = torch.zeros((input_ids_for_text.size(0), 1)).long()
    all_labels = torch.tensor([row['labels'] for i, row in data_df.iterrows()], dtype=torch.long)
    if 'filename' in data_df.columns:
        fname_vars = torch.tensor([convert_fname_to_continuous_variable(fname_to_var_key, row['filename'])
                                   for i, row in data_df.iterrows()], dtype=torch.float)

    dataset = TensorDataset(input_ids_for_text, input_ids_for_context, torch.tensor(length_of_each_maintext).long(),
                            one_where_context, all_labels)
    if 'filename' in data_df.columns:
        return dataset, fname_vars
    else:
        return dataset


def tokenize_sentence(sentence_as_string, regular_exp=None):
    if regular_exp is None:
        delimiters = [char for char in whitespace]
        regular_exp = '|'.join(map(re.escape, delimiters))
    sentence_pieces = re.split(regular_exp, sentence_as_string.strip())
    tokens_to_return = []
    for piece in sentence_pieces:
        if len(piece) == 0:
            continue
        # if it's punctuation on the end of a token, then split it off and make it its own token
        cur_ind = 0
        while cur_ind < len(piece) and piece[cur_ind] in punctuation:
            cur_ind += 1

        if cur_ind == len(piece):
            if len(piece.strip()) > 0:
                tokens_to_return.append(piece)  # the whole thing is punctuation
            continue
        else:
            piece_to_append = piece[:cur_ind].strip()
            if len(piece_to_append) > 0:
                tokens_to_return.append(piece_to_append)

        first_ind_of_non_punctuation = cur_ind

        cur_ind = len(piece) - 1
        while cur_ind > first_ind_of_non_punctuation and piece[cur_ind] in punctuation:
            cur_ind -= 1

        piece_to_append = piece[first_ind_of_non_punctuation: cur_ind + 1].strip()
        if len(piece_to_append) > 0:
            tokens_to_return.append(piece_to_append)
        if cur_ind < len(piece) - 1:
            piece_to_append = piece[cur_ind + 1:].strip()
            if len(piece_to_append) > 0:
                tokens_to_return.append(piece_to_append)

    return tokens_to_return


class SentenceIterator:
    def __init__(self, dataframe_with_data, words_to_inds, unk_token_during_embedding_training, lowercase,
                 regular_exp):
        self.df = dataframe_with_data
        self.words_to_inds = words_to_inds
        self.unk_token_string = unk_token_during_embedding_training
        self.lowercase = lowercase
        self.regular_exp = regular_exp

    def __iter__(self):
        for i, row in self.df.iterrows():
            text_in_sentence = row['text']
            tokens_in_sent = tokenize_sentence(text_in_sentence, regular_exp=self.regular_exp)
            inds_to_unk = []
            if self.lowercase:
                tokens_in_sent = [token.lower() for token in tokens_in_sent]
            for i, token in enumerate(tokens_in_sent):
                if token not in self.words_to_inds:
                    inds_to_unk.append(i)
            for ind in inds_to_unk:
                tokens_in_sent[ind] = self.unk_token_string
            if len(tokens_in_sent) > 0:
                yield tokens_in_sent

    def __call__(self, *args, **kwargs):
        return iter(self)


def develop_vocabulary(dataframe_with_data, lowercase_all_text, regular_exp=None, cutoff_for_being_in_vocab = 4):
    vocab_in_progress = {}
    total_num_sentences = 0
    for i, row in dataframe_with_data.iterrows():
        text_in_sentence = row['text']
        tokens_in_sent = tokenize_sentence(text_in_sentence, regular_exp=regular_exp)
        for token in tokens_in_sent:
            if lowercase_all_text:
                token = token.lower()
            if token not in vocab_in_progress:
                vocab_in_progress[token] = 1
            else:
                vocab_in_progress[token] += 1
        if len(tokens_in_sent) > 0:
            total_num_sentences += 1
    words_to_inds = {}
    inds_to_words = {}
    unk_token = '<unk>'
    next_available_ind = 1  # we don't start with 0 because that's reserved for padding
    for tokentype, count in vocab_in_progress.items():
        if count >= cutoff_for_being_in_vocab:
            words_to_inds[tokentype] = next_available_ind
            inds_to_words[next_available_ind] = tokentype
            next_available_ind += 1
    while unk_token in words_to_inds:
        unk_token = '<' + unk_token + '>'
    return words_to_inds, inds_to_words, unk_token, total_num_sentences


def train_and_save_word_embeddings(training_data_df, directory_for_filename,
                                   inds_to_words, total_num_sentences, unk_ind, words_to_inds, word_embedding_dimension,
                                   iterations_for_training_word_embeddings, unk_token_during_embedding_training,
                                   lowercase_all_text, regular_exp, run_check=False):
    print("Starting to put together word embeddings at " + str(datetime.datetime.now()))
    sentence_iterator = SentenceIterator(training_data_df, words_to_inds, unk_token_during_embedding_training,
                                         lowercase_all_text, regular_exp=regular_exp)
    trained_model = Word2Vec(None, iter=iterations_for_training_word_embeddings,
                             min_count=0, size=word_embedding_dimension, workers=4)
    print("Starting to build vocabulary in gensim Word2Vec model at " + str(datetime.datetime.now()))
    trained_model.build_vocab(sentence_iterator)
    print("Starting to train Word2Vec embeddings at " + str(datetime.datetime.now()))
    trained_model.train(sentence_iterator, total_examples=total_num_sentences,
                        epochs=iterations_for_training_word_embeddings)

    if run_check:
        assert len(trained_model.wv.vocab) == len(inds_to_words) + 1  # since inds_to_words doesn't include unk
        part_to_query = trained_model.wv
        print('Most similar to Faulkner:')
        try:
            print(part_to_query.most_similar(positive=['faulkner' if lowercase_all_text else 'Faulkner'],
                                             negative=[])[:10])
        except KeyError:
            print(('faulkner' if lowercase_all_text else 'Faulkner') + ' not in vocabulary')
        print('Most similar to arrest:')
        try:
            print(part_to_query.most_similar(positive=['arrest'], negative=[])[:10])
        except KeyError:
            print('arrest not in vocabulary')
        print('Most similar to provisional:')
        try:
            print(part_to_query.most_similar(positive=['provisional'], negative=[])[:10])
        except KeyError:
            print('provisional not in vocabulary')


    temp_filename = os.path.join(directory_for_filename, "_tempgensim")
    trained_model.save(temp_filename)

    print("Starting to move trained embeddings into numpy matrix at " + str(datetime.datetime.now()))
    num_vocab_words = len(inds_to_words)
    embedding_matrix = np.zeros((num_vocab_words + 2, word_embedding_dimension))
    for word_ind in tqdm(inds_to_words.keys(), total=len(inds_to_words)):
        assert word_ind != 0, 'We later assume the index 0 corresponds to padding'
        embedding_matrix[word_ind] = trained_model[inds_to_words[word_ind]]
    embedding_matrix[unk_ind] = trained_model[unk_token_during_embedding_training]
    norm_of_embeddings = np.linalg.norm(embedding_matrix, axis=1)
    norm_of_embeddings[norm_of_embeddings == 0] = 1e-13
    embedding_matrix = embedding_matrix / norm_of_embeddings[:, None]
    np.save(os.path.join(directory_for_filename, 'word2vec_embeddings.npy'), embedding_matrix)
    print('Saved trained embeddings to ' + os.path.join(directory_for_filename, 'word2vec_embeddings.npy'))

    print("Removing temporary gensim model files at " + str(datetime.datetime.now()))
    # remove gensim model files, now that embedding matrix has been saved
    if os.path.isfile(temp_filename):
        os.remove(temp_filename)
    if os.path.isfile(temp_filename + ".syn1neg.npy"):
        os.remove(temp_filename + ".syn1neg.npy")
    if os.path.isfile(temp_filename + ".wv.syn0.npy"):
        os.remove(temp_filename + ".wv.syn0.npy")


def save_vocab_index(inds_to_words, directory, unk_token, unk_ind):
    fname = os.path.join(directory, 'vocab_in_index_order.txt')
    make_directories_as_necessary(fname)
    assert unk_ind == len(inds_to_words) + 1
    with open(fname, 'w') as f:
        for i in range(1, len(inds_to_words) + 1):
            f.write(inds_to_words[i] + '\n')
        f.write(unk_token + '\n')
    print('Saved vocab index to ' + fname)


def run_classification(train_df, dev_df, lowercase_all_text, output_dir, num_labels,
                       batch_size: int, learning_rate: float, use_context, use_lstm, label_weights: List[float]=None,
                       embedding_dim=100,
                       iterations_for_pretraining_word_embeddings=50, cuda_device=-1, f1_avg='weighted', test_set=None,
                       print_results=True, also_report_binary_precrec=False, string_prefix='',
                       process_context_separately=True, pretrained_word2vec_dir=None):
    stop_if_no_improvement_after_x_epochs = 30
    iterations_for_pretraining_word_embeddings = iterations_for_pretraining_word_embeddings
    num_train_epochs = 300.0
    run_check_on_word2vec_output = True
    seed = 42
    do_eval = True
    do_train = True
    evaluate_during_training = False
    do_lower_case = lowercase_all_text
    per_gpu_train_batch_size = 3  # batch_size
    per_gpu_eval_batch_size = 3  # batch_size
    gradient_accumulation_steps = ceil(batch_size / 8)
    learning_rate = learning_rate  # 5e-5
    weight_decay = 0.0
    adam_epsilon = 1e-8
    max_grad_norm = 1.0
    max_steps = -1
    warmup_steps = 0
    logging_steps = 500
    save_steps = None  # we don't use this anymore
    eval_all_checkpoints = False
    overwrite_output_dir = False
    overwrite_cache = False
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
    ):
        will_load_preexisting_model_instead = True
    else:
        will_load_preexisting_model_instead = False

    delimiters = [char for char in whitespace]
    regular_exp = '|'.join(map(re.escape, delimiters))
    if not will_load_preexisting_model_instead:
        if pretrained_word2vec_dir is None:
            print('Training word embeddings from scratch.')
            words_to_inds, inds_to_words, unk_token, total_num_sentences_in_train = \
                develop_vocabulary(train_df, lowercase_all_text, cutoff_for_being_in_vocab=4, regular_exp=regular_exp)
            unk_ind = len(words_to_inds) + 1
            save_vocab_index(inds_to_words, output_dir, unk_token, unk_ind)
            train_and_save_word_embeddings(train_df, output_dir, inds_to_words, total_num_sentences_in_train,
                                           unk_ind, words_to_inds, embedding_dim,
                                           iterations_for_pretraining_word_embeddings, unk_token, lowercase_all_text,
                                           regular_exp=regular_exp, run_check=run_check_on_word2vec_output)
        else:
            print('Copying pretrained word embeddings from ' + pretrained_word2vec_dir + '.')
            make_directories_as_necessary(os.path.join(output_dir, 'word2vec_embeddings.npy'))
            shutil.copy(os.path.join(pretrained_word2vec_dir, 'word2vec_embeddings.npy'),
                        os.path.join(output_dir, 'word2vec_embeddings.npy'))
            shutil.copy(os.path.join(pretrained_word2vec_dir, 'vocab_in_index_order.txt'),
                        os.path.join(output_dir, 'vocab_in_index_order.txt'))
            unk_ind = 1
            words_to_inds = {}
            with open(os.path.join(output_dir, 'vocab_in_index_order.txt'), 'r') as f:
                last_thing_added = None
                for line in f:
                    token = line.strip()
                    words_to_inds[token] = unk_ind
                    last_thing_added = token
                    unk_ind += 1  # includes unk
                del words_to_inds[last_thing_added]  # don't put unk in words_to_inds
            unk_ind -= 1

        # pretrained word2vec embeddings are now saved in [directory_for_model]/word2vec_embeddings.npy
        # (corresponding vocabulary, 1-indexed, is in [directory_for_model]/vocab_in_index_order.txt

        pretrained_embeddings = np.load(os.path.join(output_dir, 'word2vec_embeddings.npy'))
        assert unk_ind == pretrained_embeddings.shape[0] - 1, 'UNK ind: ' + str(unk_ind) + \
                                                              ', size of pretrained embeddings: ' + \
                                                              str(pretrained_embeddings.shape)
        if use_lstm:
            model = LSTMNNModel(num_output_labels=num_labels,
                                pretrained_embeddings=pretrained_embeddings,
                                label_weights=label_weights,
                                process_context_separately=process_context_separately)
        else:
            model = NNModel(num_labels, pretrained_embeddings=pretrained_embeddings, label_weights=label_weights,
                            process_context_separately=process_context_separately)

        num_embeddings = unk_ind + 1
    else:
        dir_to_load = get_subdir_with_model_to_load(output_dir)
        print('Loading model from ' + dir_to_load)
        model, words_to_inds = load_in_pickled_model(dir_to_load, output_dir, cuda_device, num_labels, label_weights,
                                                     lowercase_all_text, embedding_dim=embedding_dim,
                                                     process_context_separately=process_context_separately,
                                                     use_lstm=use_lstm)

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
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if local_rank in [-1, 0] else logging.WARN,
    )
    # Set seed
    set_seed(seed, n_gpu)

    task_name = "training"
    output_mode = "classification"

    model.to(device)
    print("model is on cuda: " + str(next(model.parameters()).device))

    # Training
    if do_train and not will_load_preexisting_model_instead:
        dict_of_info_for_debugging = {'cuda_device': cuda_device, 'num_labels': num_labels,
                                      'label_weights': label_weights, 'lowercase_all_tokens': lowercase_all_text}
        train_dataset = load_and_cache_examples(train_df, lowercase_all_text=lowercase_all_text,
                                                words_to_inds=words_to_inds,
                                                includes_context_column=use_context,
                                                regular_exp=regular_exp)
        global_step, tr_loss = train(train_dataset, model, also_report_binary_precrec, f1_avg,
                                     seed, n_gpu, max_steps, local_rank, per_gpu_train_batch_size,
                                     gradient_accumulation_steps, num_train_epochs, weight_decay,
                                     learning_rate, adam_epsilon, warmup_steps, fp16,
                                     fp16_opt_level, device, max_grad_norm, logging_steps,
                                     evaluate_during_training, save_steps, per_gpu_eval_batch_size, task_name,
                                     output_mode, output_dir, dev_df, evaluate_at_end_of_epoch, save_at_end_of_epoch,
                                     metric_to_perform_better_on, higher_is_better, label_weights, num_labels,
                                     dict_of_info_for_debugging, regular_exp=regular_exp, words_to_inds=words_to_inds,
                                     lowercase_all_text=lowercase_all_text, use_context=use_context,
                                     stop_if_no_improvement_after_x_epochs=stop_if_no_improvement_after_x_epochs,
                                     num_labels=num_labels,
                                     use_lstm=use_lstm)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if do_train and True and not will_load_preexisting_model_instead:  # (local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        """model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training"""
        model.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model

        # Load a trained model and vocabulary that you have fine-tuned
        dir_to_load = get_subdir_with_model_to_load(output_dir)
        model, words_to_inds = load_in_pickled_model(dir_to_load, output_dir, cuda_device, num_labels, label_weights,
                                                     lowercase_all_text, embedding_dim,
                                                     process_context_separately=process_context_separately,
                                                     use_lstm=use_lstm)
    if will_load_preexisting_model_instead:
        dir_to_load = get_subdir_with_model_to_load(output_dir)
        model, words_to_inds = load_in_pickled_model(dir_to_load, output_dir, cuda_device, num_labels, label_weights,
                                                     lowercase_all_text, embedding_dim,
                                                     process_context_separately=process_context_separately,
                                                     use_lstm=use_lstm)

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
            model, words_to_inds = load_in_pickled_model(dir_to_load, output_dir, cuda_device, num_labels,
                                                         label_weights, lowercase_all_text, embedding_dim,
                                                         process_context_separately=process_context_separately,
                                                         use_lstm=use_lstm)
            if test_set is None:
                result = evaluate(model, also_report_binary_precrec, task_name, output_dir, local_rank,
                                  per_gpu_eval_batch_size, n_gpu, device, output_mode, dev_df,
                                  lowercase_all_text=lowercase_all_text,
                                  prefix=prefix, f1_avg_type=f1_avg, regular_exp=regular_exp,
                                  words_to_inds=words_to_inds, use_context=use_context)
            else:
                result = evaluate(model, also_report_binary_precrec, task_name, output_dir, local_rank,
                                  per_gpu_eval_batch_size, n_gpu, device, output_mode, test_set,
                                  lowercase_all_text=lowercase_all_text,
                                  prefix=prefix, f1_avg_type=f1_avg, regular_exp=regular_exp,
                                  words_to_inds=words_to_inds, use_context=use_context)
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
            print(string_prefix + 'With batch size ' + str(batch_size) + ', learning rate ' + str(learning_rate) +
                  ', and ' + ('DOUBLED' if process_context_separately else 'NO doubled') +
                  ' context features, ' + ('lstm' if use_lstm else 'feedforward') +
                  ' word2vec baseline result: accuracy is ' + str(acc) + ' and ' + f1_avg +
                  ' f1 is ' + str(f1) +
                  ' (precision is ' + str(prec) + ' and recall is ' + str(rec) + ')')
        else:
            print(string_prefix + 'With batch size ' + str(batch_size) + ', learning rate ' + str(learning_rate) +
                  ', and ' + ('DOUBLED' if process_context_separately else 'NO doubled') +
                  ' context features, ' + ('lstm' if use_lstm else 'feedforward') +
                  ' word2vec baseline result: accuracy is ' + str(acc) + ' and ' + f1_avg +
                  ' f1 is ' + str(f1))
    if also_report_binary_precrec:
        return f1, acc, model_outputs, prec, rec
    else:
        return f1, acc, model_outputs


def run_best_model_on(output_dir, test_df, num_labels, label_weights, lowercase_all_text, process_context_separately,
                      use_context, use_lstm,
                      cuda_device=-1, string_prefix='', f1_avg: str='weighted', print_results=True,
                      also_report_binary_precrec=False, embedding_dim=100):
    if cuda_device == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', cuda_device)

    dir_to_load = get_subdir_with_model_to_load(output_dir)
    print('Loading model from ' + dir_to_load)
    model, words_to_inds = load_in_pickled_model(dir_to_load, output_dir, cuda_device, num_labels, label_weights,
                                                 lowercase_all_text, embedding_dim=embedding_dim,
                                                 process_context_separately=process_context_separately,
                                                 use_lstm=use_lstm)
    result, model_outputs = evaluate(model, also_report_binary_precrec, "Classifying", output_dir,
                                     cuda_device, 4, 1, device, "classification", test_df, prefix="",
                                     f1_avg_type=f1_avg, return_model_outputs_too=True,
                                     lowercase_all_text=lowercase_all_text, words_to_inds=words_to_inds,
                                     use_context=use_context)

    if print_results:
        if also_report_binary_precrec:
            print(string_prefix + ('lstm' if use_lstm else 'feedforward') +
                  ' word2vec baseline result: accuracy is ' + str(result['acc']) + ' and ' +
                  f1_avg + ' f1 is ' +
                  str(result['f1']) + ' (precision is ' + str(result['prec']) + ' and recall is ' +
                  str(result['rec']) + ')')
        else:
            print(string_prefix + ('lstm' if use_lstm else 'feedforward') +
                  ' word2vec baseline result: accuracy is ' + str(result['acc']) + ' and ' +
                  f1_avg + ' f1 is ' +
                  str(result['f1']))
    if also_report_binary_precrec:
        return result['f1'], result['acc'], list(model_outputs), result['prec'], result['rec']
    else:
        return result['f1'], result['acc'], list(model_outputs)
