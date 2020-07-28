from simpletransformers.language_modeling import LanguageModelingModel
from torch.utils.data import Dataset
import pandas as pd
import os
import shutil
import torch
from prep_data import read_in_presplit_data
from config import binary_train_filename, binary_dev_filename, \
    binary_test_filename, binary_label_key_filename
from tqdm import tqdm


def get_gpt2_perplexity_for_every_sentence(data_as_pd_dataframe, output_file, cuda_device=-1, lowercase_all_text=True):
    if cuda_device < 0:
        use_cuda = False
    else:
        use_cuda = True
    model = LanguageModelingModel('gpt2', 'gpt2-large',
                                  use_cuda=use_cuda, cuda_device=cuda_device,
                                  args={'do_lower_case': lowercase_all_text, 'mlm': False})
    model.model = model.model.to(torch.device(f"cuda:{cuda_device}"))

    disposable_output_dir = 'scratch_output_dir/'
    if not os.path.isdir(disposable_output_dir):
        was_originally_dir = False
        os.makedirs(disposable_output_dir)
    else:
        was_originally_dir = True

    with open(output_file, 'w') as f:
        f.write('perplexity\tsentence\n')
        for i, row in tqdm(data_as_pd_dataframe.iterrows(), total=data_as_pd_dataframe.shape[0]):
            # single_example_dataset = pd.DataFrame(pd.Series([row['text']]), columns=['text'], index=[0])
            single_example_dataset = SingleItemDataset(model.tokenizer, row['text'], cuda_device)
            results = model.evaluate(single_example_dataset, disposable_output_dir, multi_label=False, verbose=False,
                                     silent=True)
            instance_perplexity = float(results['perplexity'])
            text = row['text']
            if '\n' in text or '\t' in text:
                if '"' in text:
                    f.write('\t'.join([str(instance_perplexity), '""' + text + '""']) + '\n')
                else:
                    f.write('\t'.join([str(instance_perplexity), '"' + text + '"']) + '\n')
            else:
                f.write('\t'.join([str(instance_perplexity), text]) + '\n')

    if not was_originally_dir:
        shutil.rmtree(disposable_output_dir)


class SingleItemDataset(Dataset):
    def __init__(self, tokenizer, text, cuda_device):
        self.single_example = tokenizer.build_inputs_with_special_tokens([tokenizer.encode(text)])
        self.cuda_device = cuda_device

    def __len__(self):
        return 1

    def __getitem__(self, item):
        return torch.tensor(self.single_example[0], dtype=torch.long)


if __name__ == '__main__':
    if torch.cuda.is_available():
        cuda_device = 0
    else:
        cuda_device = -1
    train_df, dev_df, test_df, num_labels = \
        read_in_presplit_data(binary_train_filename, binary_dev_filename, binary_test_filename,
                              binary_label_key_filename)
    get_gpt2_perplexity_for_every_sentence(train_df, 'training_sentence_perplexities.tsv', cuda_device=cuda_device)
