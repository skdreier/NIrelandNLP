from typing import List
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def run_classification(train_df, dev_df, regularization_weight, label_weights: List[float]=None,
                       lowercase_all_text=True, string_prefix='', print_results=True, f1_avg: str='weighted',
                       also_output_logits=False, also_report_binary_precrec=False):
    list_of_all_training_text = []
    list_of_all_training_labels = []
    if label_weights is not None:
        class_weight = {}
        for i in range(len(label_weights)):
            class_weight[i] = label_weights[i]
    else:
        class_weight = None
    for index, row in train_df.iterrows():
        text = row['text']
        if lowercase_all_text:
            text = text.lower()
        list_of_all_training_text.append(text)
        label = int(row['labels'])
        list_of_all_training_labels.append(label)
    list_of_all_dev_text = []
    list_of_all_dev_labels = []
    for index, row in dev_df.iterrows():
        text = row['text']
        if lowercase_all_text:
            text = text.lower()
        list_of_all_dev_text.append(text)
        label = int(row['labels'])
        list_of_all_dev_labels.append(label)
    cv = CountVectorizer()
    training_docs = cv.fit_transform(list_of_all_training_text)
    vocab_list = cv.get_feature_names()
    dev_docs = cv.transform(list_of_all_dev_text)

    lr_model = LogisticRegression(class_weight=class_weight, max_iter=10000, C=1/regularization_weight)
    lr_model.fit(training_docs, list_of_all_training_labels)

    predicted_labels = lr_model.predict(dev_docs)
    accuracy = float(accuracy_score(list_of_all_dev_labels, predicted_labels))
    f1 = float(f1_score(list_of_all_dev_labels, predicted_labels, average=f1_avg))
    if also_report_binary_precrec:
        prec = float(precision_score(list_of_all_dev_labels, predicted_labels, pos_label=1, average=f1_avg))
        rec = float(recall_score(list_of_all_dev_labels, predicted_labels, pos_label=1, average=f1_avg))
    if print_results:
        if also_report_binary_precrec:
            print(string_prefix + 'With regularization weight ' + str(regularization_weight) +
                  ', logistic regression result: accuracy is ' + str(accuracy) + ' and ' + f1_avg +
                  ' f1 is ' + str(f1) + ' (precision is ' + str(prec) + ' and recall is ' + str(rec) + ')')
        else:
            print(string_prefix + 'With regularization weight ' + str(regularization_weight) +
                  ', logistic regression result: accuracy is ' + str(accuracy) + ' and ' + f1_avg +
                  ' f1 is ' + str(f1))
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
