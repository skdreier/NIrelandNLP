from glob import glob
import sys


def will_accept_filename_as_source_for_hyperparams(filename, model_type, task, hyperparams):
    if model_type == 'RoBERTa':
        lowercase = hyperparams[0]
        num_sents_as_context = hyperparams[1]
        batch_size = hyperparams[2]
        learning_rate = hyperparams[3]
        if learning_rate == 3e-5 and batch_size == 136 and not lowercase:
            if num_sents_as_context == 0 or num_sents_as_context == 2 or num_sents_as_context == 4:
                if not filename.endswith('RobertaMakeupWillHopefullyComplete.txt'):
                    return False
    elif model_type == 'lstm' or model_type == 'feedforward':
        if task == 'binary':
            if not filename.endswith('NewembedsWord2vecMakeup.txt'):
                return False
        else:
            lowercase = hyperparams[0]
            pretrained_word2vec_embeddings_only_used_positive_sentences = hyperparams[5]
            if pretrained_word2vec_embeddings_only_used_positive_sentences:
                if lowercase:
                    # needs to be from the original runs
                    if not filename.endswith('OldsmallembedsWord2vecMultiway.txt'):
                        return False
                else:
                    # needs to be from the most recent runs
                    if not filename.endswith('NewsmallembedsWord2vecMultiway.txt'):
                        return False
            else:
                if not filename.endswith('NewembedsWord2vecMakeup.txt'):
                    return False
    return True


def get_best_set_of_hyperparams_for_model(model_name, task, num_sents_as_context_param=None):
    assert model_name in ['RoBERTa', 'lstm', 'feedforward', 'logreg']
    best_dev_f1_seen_so_far = 0.0
    corresponding_hparams = None
    if num_sents_as_context_param is None:
        num_sents_as_context_list = [0, 1, 2, 3, 4]
    else:
        num_sents_as_context_list = [num_sents_as_context_param]
    for num_sents_as_context in num_sents_as_context_list:
        for fname in glob('experiment_results/' + str(num_sents_as_context) +
                          '_sents_as_context/terminal_outputs/overall*'):
            all_text_lowercased = True
            in_multiway_set = None
            with open(fname, 'r') as f:
                for line in f:
                    if 'Read in existing binary data split' in line:
                        in_multiway_set = False
                    elif 'Read in existing multi-way data split' in line:
                        in_multiway_set = True
                    if line.startswith('\tWith ') and 'result: accuracy is ' in line and not line.endswith(')\n') and \
                        ('logistic regression' in line if model_name == 'logreg' else model_name in line) and \
                            (not in_multiway_set if task == 'binary' else in_multiway_set):
                        # this is a dev-set-performance-reporting line including its hyperparams, which we collect
                        assert in_multiway_set is not None
                        f1 = float(line[line.index('f1 is ') + len('f1 is '):])
                        if f1 > best_dev_f1_seen_so_far:
                            line = line[len('\tWith '):]
                            line = line[:line.rfind(',')]
                            hparams = None
                            if model_name == 'RoBERTa':
                                # batch size 16 and learning rate 1e-05
                                batch_size = int(line[line.index('batch size ') + len('batch size '):
                                                      line.index(' and')])
                                learning_rate = float(line[line.rfind(' ') + 1:])
                                hparams = (all_text_lowercased, num_sents_as_context, batch_size, learning_rate)
                            elif model_name == 'lstm':
                                # batch size 16, learning rate 5e-05, and NO doubled context features
                                batch_size = int(line[line.index('batch size ') + len('batch_size '): line.index(', ')])
                                line = line[line.index(', ') + 1:]
                                learning_rate = line[:line.index(', ')]
                                learning_rate = float(learning_rate[learning_rate.rfind(' ') + 1:])
                                doubled_context_features = (' NO doubled context features' not in line)
                                if (task != 'binary' and fname.endswith('OldsmallembedsWord2vecMultiway.txt')) or \
                                        fname.endswith('NewsmallembedsWord2vecMultiway.txt'):
                                    assert task != 'binary'
                                    word2vecembeds_only_pretrained_on_positive_sents = True
                                else:
                                    word2vecembeds_only_pretrained_on_positive_sents = False
                                hparams = (all_text_lowercased, num_sents_as_context, batch_size, learning_rate,
                                           doubled_context_features, word2vecembeds_only_pretrained_on_positive_sents)
                            elif model_name == 'feedforward':
                                # batch size 32, learning rate 0.001, and NO doubled context features
                                batch_size = int(line[line.index('batch size ') + len('batch_size '): line.index(', ')])
                                line = line[line.index(', ') + 1:]
                                learning_rate = line[:line.index(', ')]
                                learning_rate = float(learning_rate[learning_rate.rfind(' ') + 1:])
                                doubled_context_features = (' NO doubled context features' not in line)
                                if (task != 'binary' and fname.endswith('OldsmallembedsWord2vecMultiway.txt')) or \
                                        fname.endswith('NewsmallembedsWord2vecMultiway.txt'):
                                    assert task != 'binary'
                                    word2vecembeds_only_pretrained_on_positive_sents = True
                                else:
                                    word2vecembeds_only_pretrained_on_positive_sents = False
                                hparams = (all_text_lowercased, num_sents_as_context, batch_size, learning_rate,
                                           doubled_context_features, word2vecembeds_only_pretrained_on_positive_sents)
                            elif model_name == 'logreg':
                                # regularization weight 0.0001 and NO doubled context features
                                reg_weight = float(line[line.index('regularization weight ') +
                                                        len('regularization weight '):
                                                        line.index(' and')])
                                doubled_context_features = (' NO doubled context features' not in line)
                                hparams = (all_text_lowercased, num_sents_as_context, reg_weight,
                                           doubled_context_features)
                            assert hparams is not None
                            if will_accept_filename_as_source_for_hyperparams(fname, model_name, task, hparams):
                                corresponding_hparams = hparams
                                best_dev_f1_seen_so_far = f1
                            else:
                                continue
    for num_sents_as_context in num_sents_as_context_list:
        for fname in glob('experiment_results/' + str(num_sents_as_context) +
                          '_sents_as_context_CASED/terminal_outputs/overall*'):
            all_text_lowercased = False
            in_multiway_set = None
            with open(fname, 'r') as f:
                for line in f:
                    if 'Read in existing binary data split' in line:
                        in_multiway_set = False
                    elif 'Read in existing multi-way data split' in line:
                        in_multiway_set = True
                    if line.startswith('\tWith ') and 'result: accuracy is ' in line and not line.endswith(')\n') and \
                        ('logistic regression' in line if model_name == 'logreg' else model_name in line) and \
                            (not in_multiway_set if task == 'binary' else in_multiway_set):
                        # this is a dev-set-performance-reporting line including its hyperparams, which we collect
                        assert in_multiway_set is not None
                        f1 = float(line[line.index('f1 is ') + len('f1 is '):])
                        if f1 > best_dev_f1_seen_so_far:
                            line = line[len('\tWith '):]
                            line = line[:line.rfind(',')]
                            hparams = None
                            if model_name == 'RoBERTa':
                                # batch size 16 and learning rate 1e-05
                                batch_size = int(line[line.index('batch size ') + len('batch size '):
                                                      line.index(' and')])
                                learning_rate = float(line[line.rfind(' ') + 1:])
                                hparams = (all_text_lowercased, num_sents_as_context, batch_size, learning_rate)
                            elif model_name == 'lstm':
                                # batch size 16, learning rate 5e-05, and NO doubled context features
                                batch_size = int(line[line.index('batch size ') + len('batch_size '): line.index(', ')])
                                line = line[line.index(', ') + 1:]
                                learning_rate = line[:line.index(', ')]
                                learning_rate = float(learning_rate[learning_rate.rfind(' ') + 1:])
                                doubled_context_features = (' NO doubled context features' not in line)
                                if (task != 'binary' and fname.endswith('OldsmallembedsWord2vecMultiway.txt')) or \
                                        fname.endswith('NewsmallembedsWord2vecMultiway.txt'):
                                    assert task != 'binary'
                                    word2vecembeds_only_pretrained_on_positive_sents = True
                                else:
                                    word2vecembeds_only_pretrained_on_positive_sents = False
                                hparams = (all_text_lowercased, num_sents_as_context, batch_size, learning_rate,
                                           doubled_context_features, word2vecembeds_only_pretrained_on_positive_sents)
                            elif model_name == 'feedforward':
                                # batch size 32, learning rate 0.001, and NO doubled context features
                                batch_size = int(line[line.index('batch size ') + len('batch_size '): line.index(', ')])
                                line = line[line.index(', ') + 1:]
                                learning_rate = line[:line.index(', ')]
                                learning_rate = float(learning_rate[learning_rate.rfind(' ') + 1:])
                                doubled_context_features = (' NO doubled context features' not in line)
                                if (task != 'binary' and fname.endswith('OldsmallembedsWord2vecMultiway.txt')) or \
                                        fname.endswith('NewsmallembedsWord2vecMultiway.txt'):
                                    assert task != 'binary'
                                    word2vecembeds_only_pretrained_on_positive_sents = True
                                else:
                                    word2vecembeds_only_pretrained_on_positive_sents = False
                                hparams = (all_text_lowercased, num_sents_as_context, batch_size, learning_rate,
                                           doubled_context_features, word2vecembeds_only_pretrained_on_positive_sents)
                            elif model_name == 'logreg':
                                # regularization weight 0.0001 and NO doubled context features
                                reg_weight = float(line[line.index('regularization weight ') +
                                                        len('regularization weight '):
                                                        line.index(' and')])
                                doubled_context_features = (' NO doubled context features' not in line)
                                hparams = (all_text_lowercased, num_sents_as_context, reg_weight,
                                           doubled_context_features)
                            assert hparams is not None
                            if will_accept_filename_as_source_for_hyperparams(fname, model_name, task, hparams):
                                corresponding_hparams = hparams
                                best_dev_f1_seen_so_far = f1
                            else:
                                continue
    assert corresponding_hparams is not None
    return corresponding_hparams, best_dev_f1_seen_so_far


def hparams_match(desired_set_of_hparams, part_of_line, model_type):
    if model_type == 'logreg':
        # regularization weight, doubled context features
        assert part_of_line.startswith('regularization weight ')
        part_of_line = part_of_line[len('regularization weight '):]
        reg_weight = float(part_of_line[:part_of_line.index(' ')])
        if reg_weight != desired_set_of_hparams[2]:
            return False
        doubled_feats = 'NO doubled features' not in part_of_line
        if (doubled_feats and not desired_set_of_hparams[3]) or (desired_set_of_hparams[3] and not doubled_feats):
            return False
        return True
    elif model_type == 'RoBERTa':
        # batch size, learning rate
        assert part_of_line.startswith('lr ')
        part_of_line = part_of_line[len('lr '):]
        learning_rate = float(part_of_line[:part_of_line.index(' ')])
        if learning_rate != desired_set_of_hparams[3]:
            return False
        batch_size = int(part_of_line[part_of_line.index('batch size ') + len('batch size '):])
        if batch_size != desired_set_of_hparams[2]:
            return False
        return True
    elif model_type == 'lstm' or model_type == 'feedforward':
        # batch size, learning rate, doubled context features
        assert part_of_line.startswith('lr ')
        learning_rate = float(part_of_line[part_of_line.index('lr ') + len('lr '): part_of_line.index(', ')])
        if learning_rate != desired_set_of_hparams[3]:
            return False
        batch_size = int(part_of_line[part_of_line.index('batch size ') + len('batch size '):
                                      part_of_line.index(', and')])
        if batch_size != desired_set_of_hparams[2]:
            return False
        doubled_feats = 'NO doubled features' not in part_of_line
        if (doubled_feats and not desired_set_of_hparams[4]) or (desired_set_of_hparams[4] and not doubled_feats):
            return False
        return True


def print_test_performance_for_best_model(model_name, task, num_sents_as_context_param=None):
    assert model_name in ['RoBERTa', 'lstm', 'feedforward', 'logreg']
    if task != 'binary':
        task = 'multiway'
    best_dev_params, best_dev_f1 = \
        get_best_set_of_hyperparams_for_model(model_name, task, num_sents_as_context_param=num_sents_as_context_param)
    if best_dev_params[0]:
        fname_end = '_sents_as_context/terminal_outputs/overall*'
    else:
        fname_end = '_sents_as_context_CASED/terminal_outputs/overall*'
    num_sents_as_context_list = [best_dev_params[1]]

    print('Best dev F1 of any ' + model_name + ' model: ' + str(best_dev_f1))
    print('Corresponding best hparams: ' + str(best_dev_params))

    if model_name == 'lstm' or model_name == 'feedforward':
        test_line_start = 'For ' + task + ' case, best ' + model_name + ' word2vec baseline' + ' model had '
    elif model_name == 'logreg':
        test_line_start = 'For ' + task + ' case, best ' + 'baseline ' + model_name + ' model had '
    else:
        test_line_start = 'For ' + task + ' case, best ' + model_name + ' model had '

    for num_sents_as_context in num_sents_as_context_list:
        for fname in glob('experiment_results/' + str(num_sents_as_context) + fname_end):
            passed_line_indicating_test_perf_for_our_model_coming_up = False
            with open(fname, 'r') as f:
                in_multiway_set = None
                for line in f:
                    if 'Read in existing binary data split' in line:
                        in_multiway_set = False
                        passed_line_indicating_test_perf_for_our_model_coming_up = False
                    elif 'Read in existing multi-way data split' in line:
                        in_multiway_set = True
                        passed_line_indicating_test_perf_for_our_model_coming_up = False

                    if line.startswith(test_line_start):
                        # check whether the hyperparams in the line match the ones we're looking for
                        if model_name == 'logreg':
                            line = line[len(test_line_start): line.index(', and achieved the following performance')]
                            passed_line_indicating_test_perf_for_our_model_coming_up = hparams_match(best_dev_params,
                                                                                                     line, model_name)
                        else:
                            line = line[len(test_line_start): line.index('. Performance:')]
                            passed_line_indicating_test_perf_for_our_model_coming_up = hparams_match(best_dev_params,
                                                                                                     line, model_name)

                    if passed_line_indicating_test_perf_for_our_model_coming_up:
                        if model_name == 'logreg':
                            if line.startswith('(Test set) With regularization weight ') and \
                                    (not in_multiway_set if task == 'binary' else in_multiway_set):
                                assert in_multiway_set is not None
                                if will_accept_filename_as_source_for_hyperparams(fname, model_name, task,
                                                                                  best_dev_params):
                                    # this is the line with our test set performance
                                    print(line + '\t(From ' + fname + ')')
                                passed_line_indicating_test_perf_for_our_model_coming_up = False
                        else:
                            if line.startswith('Test ' + model_name) and ' result: accuracy is ' in line and \
                                    line.endswith(')\n') and \
                                    (not in_multiway_set if task == 'binary' else in_multiway_set):
                                assert in_multiway_set is not None
                                if will_accept_filename_as_source_for_hyperparams(fname, model_name, task,
                                                                                  best_dev_params):
                                    # this is the line with our test set performance
                                    print(line + '\t(From ' + fname + ')')
                                passed_line_indicating_test_perf_for_our_model_coming_up = False


if __name__ == '__main__':
    model_name = sys.argv[1].strip()
    task = sys.argv[2].strip()
    if len(list(sys.argv)) < 4:
        num_sents_as_context = None
    else:
        num_sents_as_context = sys.argv[3].strip()
        if num_sents_as_context.isdigit():
            num_sents_as_context = int(num_sents_as_context)
        else:
            num_sents_as_context = None
    print_test_performance_for_best_model(model_name, task, num_sents_as_context_param=num_sents_as_context)
