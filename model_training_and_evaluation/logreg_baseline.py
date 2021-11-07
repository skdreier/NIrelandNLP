from typing import List
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from scipy.sparse import hstack


def run_classification(train_df, dev_df, regularization_weight, label_weights: List[float]=None,
                       lowercase_all_text=True, string_prefix='', print_results=True, f1_avg: str='weighted',
                       also_output_logits=False, also_report_binary_precrec=False, double_context_features=False,
                       use_context=None):
    assert use_context is not None
    list_of_all_training_text = []
    list_of_all_training_contexts = []
    list_of_all_training_labels = []
    if label_weights is not None:
        class_weight = {}
        for i in range(len(label_weights)):
            class_weight[i] = label_weights[i]
    else:
        class_weight = None
    for index, row in train_df.iterrows():
        if use_context:
            if not double_context_features:
                text = row['contextbefore'] + ' ' + row['text']
            else:
                text = row['text']
                context = row['contextbefore']
        else:
            text = row['text']
        if lowercase_all_text:
            text = text.lower()
            if use_context and double_context_features:
                context = context.lower()
        if use_context and double_context_features:
            list_of_all_training_contexts.append(context)
        list_of_all_training_text.append(text)
        label = int(row['labels'])
        list_of_all_training_labels.append(label)

    list_of_all_dev_contexts = []
    list_of_all_dev_text = []
    list_of_all_dev_labels = []
    for index, row in dev_df.iterrows():
        if use_context:
            if not double_context_features:
                text = row['contextbefore'] + ' ' + row['text']
            else:
                text = row['text']
                context = row['contextbefore']
        else:
            text = row['text']
        if lowercase_all_text:
            text = text.lower()
            if use_context and double_context_features:
                context = context.lower()
        if use_context and double_context_features:
            list_of_all_dev_contexts.append(context)
        list_of_all_dev_text.append(text)
        label = int(row['labels'])
        list_of_all_dev_labels.append(label)
    cv = CountVectorizer()
    context_cv = CountVectorizer()
    training_docs = cv.fit_transform(list_of_all_training_text)
    vocab_list = cv.get_feature_names()
    dev_docs = cv.transform(list_of_all_dev_text)
    if use_context and double_context_features:
        training_contexts = context_cv.fit_transform(list_of_all_training_contexts)
        context_vocab_list = cv.get_feature_names()
        dev_contexts = context_cv.transform(list_of_all_dev_contexts)
        training_docs = hstack([training_contexts, training_docs])
        dev_docs = hstack([dev_contexts, dev_docs])

    lr_model = LogisticRegression(class_weight=class_weight, max_iter=10000, C=1/regularization_weight)
    lr_model.fit(training_docs, list_of_all_training_labels)

    predicted_labels = lr_model.predict(dev_docs)
    accuracy = float(accuracy_score(list_of_all_dev_labels, predicted_labels))
    f1 = float(f1_score(list_of_all_dev_labels, predicted_labels, average=f1_avg))
    if also_report_binary_precrec:
        prec = float(precision_score(list_of_all_dev_labels, predicted_labels, average=f1_avg))
        rec = float(recall_score(list_of_all_dev_labels, predicted_labels, average=f1_avg))
    if print_results:
        if also_report_binary_precrec:
            if double_context_features:
                print(string_prefix + 'With regularization weight ' + str(regularization_weight) +
                      ' and DOUBLED context features, logistic regression result: accuracy is ' + str(accuracy) +
                      ' and ' + f1_avg + ' f1 is ' + str(f1) + ' (precision is ' + str(prec) + ' and recall is ' +
                      str(rec) + ')')
            else:
                print(string_prefix + 'With regularization weight ' + str(regularization_weight) +
                      ' and NO doubled context features, logistic regression result: accuracy is ' + str(accuracy) +
                      ' and ' + f1_avg + ' f1 is ' + str(f1) + ' (precision is ' + str(prec) + ' and recall is ' +
                      str(rec) + ')')
        else:
            if double_context_features:
                print(string_prefix + 'With regularization weight ' + str(regularization_weight) +
                      ' and DOUBLED context features, logistic regression result: accuracy is ' + str(accuracy) +
                      ' and ' + f1_avg + ' f1 is ' + str(f1))
            else:
                print(string_prefix + 'With regularization weight ' + str(regularization_weight) +
                      ' and NO doubled context features, logistic regression result: accuracy is ' + str(accuracy) +
                      ' and ' + f1_avg + ' f1 is ' + str(f1))
    if not also_output_logits:
        if also_report_binary_precrec:
            return f1, accuracy, list_of_all_dev_labels, list(predicted_labels), prec, rec
        else:
            return f1, accuracy, list_of_all_dev_labels, list(predicted_labels)
    else:
        # get logits
        output_logits = lr_model.predict_log_proba(dev_docs)
        if also_report_binary_precrec:
            return f1, accuracy, list_of_all_dev_labels, list(predicted_labels), output_logits, prec, rec
        else:
            return f1, accuracy, list_of_all_dev_labels, list(predicted_labels), output_logits
