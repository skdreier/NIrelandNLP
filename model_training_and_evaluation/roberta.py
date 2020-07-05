from typing import List
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import accuracy_score, f1_score
import torch
import os


def load_in_pickled_model(filename, cuda_device, num_labels, label_weights, lowercase_all_text):
    model = ClassificationModelWithSavingAndLoading('roberta', filename, num_labels=num_labels,
                                                    weight=label_weights, use_cuda=False, cuda_device=cuda_device,
                                                    args={'reprocess_input_data': True,
                                                          'do_lower_case': lowercase_all_text})
    """if cuda_device != -1 and torch.cuda.is_available():
        if cuda_device == -1:
            model.device = torch.device("cuda")
        else:
            model.device = torch.device(f"cuda:{cuda_device}")
    model._move_model_to_device()"""
    return model


class ClassificationModelWithSavingAndLoading(ClassificationModel):
    def save_in_pickle_format(self, filename):
        pass
        #self.device = 'cpu'
        #self._move_model_to_device()


def run_classification(train_df, dev_df, num_labels, output_dir, batch_size: int, learning_rate: float,
                       label_weights: List[float]=None, string_prefix='',
                       lowercase_all_text=True, cuda_device=-1, f1_avg='weighted', test_set=None,
                       print_results=True):
    if cuda_device < 0:
        use_cuda = False
    else:
        use_cuda = True
    if not output_dir.endswith('/'):
        output_dir += '/'
    model = ClassificationModelWithSavingAndLoading('roberta', 'roberta-base', num_labels=num_labels,
                                                    weight=label_weights, use_cuda=use_cuda, cuda_device=cuda_device,
                                                    args={'reprocess_input_data': True,
                                                          'do_lower_case': lowercase_all_text})
    model.train_model(train_df, output_dir=output_dir,
                      f1=(lambda labels, preds: f1_score(labels, preds, average=f1_avg)),
                      eval_df=dev_df, args={'evaluate_during_training': True, 'num_train_epochs': 10,
                                            'use_early_stopping': True,
                                            'train_batch_size': batch_size,
                                            'learning_rate': learning_rate,
                                            'early_stopping_metric': 'f1',
                                            'early_stopping_delta': 0.0})
    if test_set is not None:
        result, model_outputs, wrong_predictions = \
            model.eval_model(test_set, acc=accuracy_score,
                             f1=(lambda labels, preds: f1_score(labels, preds, average=f1_avg)))
    else:
        result, model_outputs, wrong_predictions = \
            model.eval_model(dev_df, acc=accuracy_score,
                             f1=(lambda labels, preds: f1_score(labels, preds, average=f1_avg)))

    if print_results:
        print(string_prefix + 'With batch size ' + str(batch_size) + ' and learning rate ' + str(learning_rate) +
              ', RoBERTa result: accuracy is ' + str(result['acc']) + ' and ' + f1_avg + ' f1 is ' +
              str(result['f1']))
    return result['f1'], result['acc'], list(model_outputs)


# from https://stackoverflow.com/questions/800197/how-to-get-all-of-the-immediate-subdirectories-in-python
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def run_best_model_on(output_dir, test_df, num_labels, label_weights, lowercase_all_text,
                      cuda_device=-1, string_prefix='', f1_avg: str='weighted'):
    if not output_dir.endswith('/'):
        output_dir += '/'

    # since we used early stopping during training, we need to find the name of the directory that represents
    # the last saved model
    checkpoints = get_immediate_subdirectories(output_dir)
    last_epoch_found = -1
    inds_of_last_epoch = []
    for i, checkpoint in enumerate(checkpoints):
        if checkpoint.count('-') < 3:
            continue  # not in the right format to qualify
        epoch = int(checkpoint[checkpoint.rfind('-') + 1:])
        if epoch > last_epoch_found:
            last_epoch_found = epoch
            inds_of_last_epoch = [i]
        elif epoch == last_epoch_found:
            inds_of_last_epoch.append(i)

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

    model = load_in_pickled_model(output_dir, cuda_device, num_labels, label_weights, lowercase_all_text)
    result, model_outputs, wrong_predictions = \
        model.eval_model(test_df, acc=accuracy_score,
                         f1=(lambda labels, preds: f1_score(labels, preds, average=f1_avg)))
    print(string_prefix + 'RoBERTa result: accuracy is ' + str(result['acc']) +
          ' and ' + f1_avg + ' f1 is ' + str(result['f1']))
    return result['f1'], result['acc'], list(model_outputs)
