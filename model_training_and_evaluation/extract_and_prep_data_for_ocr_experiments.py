from prep_data import extract_and_tag_next_document, get_sentence_split_inds_spacy
from util import make_directories_as_necessary
from glob import glob
from Bio import pairwise2
from ocr_scoring import match_scorer, replace_filler_chars_with_none, \
    align_long_sequences_by_frankensteining_shorter_alignments, print_alignments, \
    get_index_of_nth_nonnull_char_in_list, check_alignment_list_has_same_sequence_of_characters_as_original_doc
from string import whitespace
import json


directory_to_use = '../orig_text_data/samples_for_ocr/'


def get_tagdocs_from_file(full_doc_fname):
    previously_extracted_header = None
    tags_to_documents = {}
    with open(full_doc_fname, 'r', encoding='utf-8-sig') as f:
        keep_going = True
        while keep_going:
            document, tag, previously_extracted_header = \
                extract_and_tag_next_document(f, previously_extracted_header=previously_extracted_header)
            if document is None:
                keep_going = False
            else:
                tags_to_documents[tag] = document
    return tags_to_documents


def write_docs_to_uncategorized_dir():
    # tags are in format (simplified_file_name_string, simplified_image_name_string) (e.g. ('DEFE_13_919', 'IMG_1944'))
    tags_to_docs = get_tagdocs_from_file('../orig_text_data/internment.txt')
    uncategorized_dir = directory_to_use + 'uncategorized_so_far/'
    make_directories_as_necessary(uncategorized_dir + 'throwaway')
    for tag in tags_to_docs.keys():
        # we only want to collect documents that have images with a number matching 1xxx
        image_num = tag[1]
        image_num = image_num[image_num.index('_') + 1:]
        if len(image_num) == 4 and image_num[0] == '1':
            # save the text from this in its own file
            with open(uncategorized_dir + tag[1] + '_' + tag[0] + '.txt', 'w') as f:
                f.write(tags_to_docs[tag])


def split_gold_into_sents_using_spacy():
    # first identify all gold files
    all_gold_filenames = []
    corr_filenames_to_write_to = []
    for filename in glob(directory_to_use + '**/*.txt', recursive=True):
        just_the_directories = filename[: filename.rfind('/')]
        if just_the_directories.endswith('/gold'):
            all_gold_filenames.append(filename)
            corr_filename = just_the_directories + '_splitsents' + filename[filename.rfind('/'): filename.rfind('.')] +\
                            '.json'
            make_directories_as_necessary(corr_filename)
            corr_filenames_to_write_to.append(corr_filename)

    # now split the contents of gold files into sentences using spacy
    for gold_filename, corr_filename in zip(all_gold_filenames, corr_filenames_to_write_to):
        gold_text = ''
        with open(gold_filename, 'r') as f:
            for line in f:
                gold_text += line
        #gold_text = gold_text.replace('\n', ' ')

        sentence_ends = get_sentence_split_inds_spacy(gold_text)
        sent_start = 0
        sents = []
        for i in range(len(sentence_ends)):
            sents.append(gold_text[sent_start: sentence_ends[i]])
            sent_start = sentence_ends[i]
        with open(corr_filename, 'w') as f:
            json.dump(sents, f)
        # to spot-check whether sentences seem reasonable
        print('\n==================================\n'.join(sents) + '\n==================================\n')


def split_ocr_into_sents_by_aligning_with_gold_sent_splits():
    # first identify all gold files
    all_goldsplit_filenames = []
    corr_split_filenames_to_write_to = []
    corr_unsplit_ocr_filenames = []
    for filename in glob(directory_to_use + '**/*.json', recursive=True):
        just_the_directories = filename[: filename.rfind('/')]
        if just_the_directories.endswith('/gold_splitsents'):
            all_goldsplit_filenames.append(filename)
            corr_split_filename = just_the_directories[:just_the_directories.rfind('/')] + '/ocr_splitsents' + \
                            filename[filename.rfind('/'):]
            make_directories_as_necessary(corr_split_filename)
            corr_split_filenames_to_write_to.append(corr_split_filename)

            corr_unsplit_ocr_filename = just_the_directories[:just_the_directories.rfind('/')] + \
                                        filename[filename.rfind('/'): filename.rfind('.')] + '.txt'
            corr_unsplit_ocr_filenames.append(corr_unsplit_ocr_filename)

    for goldsplit_filename, corr_split_filename, corr_unsplit_ocr_filename in \
            zip(all_goldsplit_filenames, corr_split_filenames_to_write_to, corr_unsplit_ocr_filenames):
        with open(goldsplit_filename, 'r') as f:
            gold_sents = json.load(f)
        full_gold_doc = ''.join(gold_sents)
        total_num_nonnull_chars_before_end_of_sents = []
        total = 0
        for i in range(len(gold_sents)):
            total += len(gold_sents[i])
            total_num_nonnull_chars_before_end_of_sents.append(total)

        full_ocr_doc = ''
        with open(corr_unsplit_ocr_filename, 'r') as f:
            for line in f:
                full_ocr_doc += line

        whitespace_set = set([char for char in whitespace])
        penalize_for_case_errors_in_match = True

        if max(len(full_gold_doc), len(full_ocr_doc)) <= 600:
            alignments = \
                pairwise2.align.globalcs(list(full_gold_doc), list(full_ocr_doc),
                                         match_fn=lambda x, y: match_scorer(x, y, whitespace_set,
                                                                            penalize_for_case_errors_in_match),
                                         open=-1.0, extend=-.5, gap_char=[''])
            alignments = alignments[0]
            gold_aligned = [char for char in alignments.seqA]
            guessed_aligned = [char for char in alignments.seqB]
            guessed_aligned = replace_filler_chars_with_none(full_ocr_doc, guessed_aligned)
            gold_aligned = replace_filler_chars_with_none(full_gold_doc, gold_aligned)
        else:
            gold_aligned, guessed_aligned = align_long_sequences_by_frankensteining_shorter_alignments(full_gold_doc,
                                                                                                       full_ocr_doc)
        print_alignments(gold_aligned, guessed_aligned)
        check_alignment_list_has_same_sequence_of_characters_as_original_doc(gold_aligned, full_gold_doc, halfwindow=10)
        check_alignment_list_has_same_sequence_of_characters_as_original_doc(guessed_aligned, full_ocr_doc, halfwindow=10)
        first_ind_of_aligned_sentence = 0
        ocr_sents = []
        for i in range(len(total_num_nonnull_chars_before_end_of_sents)):
            cutoff = get_index_of_nth_nonnull_char_in_list(gold_aligned, total_num_nonnull_chars_before_end_of_sents[i])
            ocr_sent = guessed_aligned[first_ind_of_aligned_sentence: cutoff]
            for j in range(len(ocr_sent)):
                if ocr_sent[j] is None:
                    ocr_sent[j] = ''
            ocr_sents.append(''.join(ocr_sent))
            first_ind_of_aligned_sentence = cutoff
        with open(corr_split_filename, 'w') as f:
            json.dump(ocr_sents, f)


if __name__ == '__main__':
    if len(directory_to_use) == 0:
        directory_to_use += '.'
    if not directory_to_use.endswith('/'):
        directory_to_use += '/'
    make_directories_as_necessary(directory_to_use + 'throwaway')
    #write_docs_to_uncategorized_dir()  # run this to pull out docs that you'll then need to manually sort + transcribe
    # then run these next two lines after you've transcribed some gold docs
    split_gold_into_sents_using_spacy()
    split_ocr_into_sents_by_aligning_with_gold_sent_splits()
