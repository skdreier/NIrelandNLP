from typing import List
import matplotlib.pyplot as plt
import os
import pandas as pd


def read_in_existing_data(filename):
    similarity_scores = []
    corr_tag_pairs = []
    with open(filename, 'r') as f:
        for line in f:
            fields = line.strip().split(': ')
            similarity_scores.append(float(fields[0]))
            tags = fields[1].split(' to ')
            corr_tag_pairs.append((tags[0], tags[1]))
    return similarity_scores, corr_tag_pairs


def make_histogram(similarity_scores):
    plt.hist(similarity_scores, bins=[0.1*val for val in range(11)])
    plt.show()


def cluster_tags_above_a_threshold(threshold, sim_scores, corr_tag_pairs, existing_cluster_id_to_alltagsincluster=None):
    # treat anything above the theshold as signifying a link
    tag_to_clusterid = {}
    clusterid_to_alltagsincluster = {}
    if existing_cluster_id_to_alltagsincluster is not None:
        for clusterid, alltagsincluster in existing_cluster_id_to_alltagsincluster.items():
            clusterid_to_alltagsincluster[clusterid] = alltagsincluster
            for tag in alltagsincluster:
                tag_to_clusterid[tag] = clusterid

    for i, sim_score in enumerate(sim_scores[1:]):
        assert sim_score <= sim_scores[i], str(sim_score) + ', ' + str(sim_scores[i])
    for i, sim_score in enumerate(sim_scores):
        if sim_score < threshold:
            break
        tags_to_link = corr_tag_pairs[i]
        if tags_to_link[0] in tag_to_clusterid and tags_to_link[1] in tag_to_clusterid:
            pass
            # merge these two clusters
            if tag_to_clusterid[tags_to_link[0]] != tag_to_clusterid[tags_to_link[1]]:
                cluster_id_to_merge_under = tag_to_clusterid[tags_to_link[0]]
                cur_tag_for_cluster1 = tag_to_clusterid[tags_to_link[1]]
                cluster_to_append_to = clusterid_to_alltagsincluster[cluster_id_to_merge_under]
                tags_to_append = clusterid_to_alltagsincluster[cur_tag_for_cluster1]
                assert tags_to_link[1] in tags_to_append
                assert cur_tag_for_cluster1 in tags_to_append
                for tag in tags_to_append:
                    cluster_to_append_to.append(tag)
                    tag_to_clusterid[tag] = cluster_id_to_merge_under
                del clusterid_to_alltagsincluster[cur_tag_for_cluster1]
        elif tags_to_link[0] in tag_to_clusterid:
            tag_to_clusterid[tags_to_link[1]] = tag_to_clusterid[tags_to_link[0]]
            clusterid_to_alltagsincluster[tag_to_clusterid[tags_to_link[1]]].append(tags_to_link[1])
        elif tags_to_link[1] in tag_to_clusterid:
            tag_to_clusterid[tags_to_link[0]] = tag_to_clusterid[tags_to_link[1]]
            clusterid_to_alltagsincluster[tag_to_clusterid[tags_to_link[0]]].append(tags_to_link[0])
        else:
            # new cluster
            tag_to_clusterid[tags_to_link[0]] = tags_to_link[0]
            tag_to_clusterid[tags_to_link[1]] = tags_to_link[0]
            clusterid_to_alltagsincluster[tags_to_link[0]] = [tags_to_link[0], tags_to_link[1]]
    return clusterid_to_alltagsincluster


def calculate_number_of_docs_marked_as_duplicates_using_threshold(threshold, sim_scores, corr_tag_pairs):
    clusterid_to_alltagsincluster = cluster_tags_above_a_threshold(threshold, sim_scores, corr_tag_pairs)
    denom_set = set()
    for tag1, tag2 in corr_tag_pairs:
        denom_set.add(tag1)
        denom_set.add(tag2)
    denom = len(denom_set)
    num_clusters = len(clusterid_to_alltagsincluster)
    num_duplicates = 0
    for clusterlist in clusterid_to_alltagsincluster.values():
        num_duplicates += (len(clusterlist) - 1)
    print(str(num_duplicates) + ' / ' + str(denom) + ' documents are duplicates (in ' + str(num_clusters) +
          ' separate clusters)')


def get_list_of_filenames_to_cluster_together(filename_with_scores='document_bow_cosinesimilarity.txt', threshold=.88,
                                              train_df_to_align_split_with: pd.DataFrame = None,
                                              dev_df_to_align_split_with: pd.DataFrame = None,
                                              test_df_to_align_split_with: pd.DataFrame = None,
                                              scorelist_for_clustering=None,
                                              corr_tagpair_list=None):
    if scorelist_for_clustering is None or corr_tagpair_list is None:
        assert filename_with_scores is not None and os.path.isfile(filename_with_scores)
        similarity_scores, tag_pairs = read_in_existing_data(filename_with_scores)
        print('Read in data from ' + filename_with_scores)
    else:
        similarity_scores, tag_pairs = scorelist_for_clustering, corr_tagpair_list
    id_to_tagsincluster = cluster_tags_above_a_threshold(threshold, similarity_scores, tag_pairs)
    if train_df_to_align_split_with is not None:
        print('Adding train-split information to clustering')
        needs_to_be_merged_train = get_list_of_filenames_to_merge_together(train_df_to_align_split_with)
        needs_to_be_merged_dev = get_list_of_filenames_to_merge_together(dev_df_to_align_split_with)
        needs_to_be_merged_test = get_list_of_filenames_to_merge_together(test_df_to_align_split_with)

        id_to_tagsincluster = \
            cluster_tags_above_a_threshold(0.5, [1.0] * len(needs_to_be_merged_train), needs_to_be_merged_train,
                                           existing_cluster_id_to_alltagsincluster=id_to_tagsincluster)
        train_id_to_docs = None
        train_ids = set([needs_to_be_merged_train[i][1] for i in range(len(needs_to_be_merged_train))])
        train_ids.add(needs_to_be_merged_train[0][0])
        for id in id_to_tagsincluster:
            if id in train_ids:
                assert train_id_to_docs is None
                train_id_to_docs = id

        id_to_tagsincluster = \
            cluster_tags_above_a_threshold(0.5, [1.0] * len(needs_to_be_merged_dev), needs_to_be_merged_dev,
                                           existing_cluster_id_to_alltagsincluster=id_to_tagsincluster)
        dev_id_to_docs = None
        dev_ids = set([needs_to_be_merged_dev[i][1] for i in range(len(needs_to_be_merged_dev))])
        dev_ids.add(needs_to_be_merged_dev[0][0])
        for id in id_to_tagsincluster:
            if id in dev_ids:
                assert dev_id_to_docs is None
                dev_id_to_docs = id

        id_to_tagsincluster = \
            cluster_tags_above_a_threshold(0.5, [1.0] * len(needs_to_be_merged_test), needs_to_be_merged_test,
                                           existing_cluster_id_to_alltagsincluster=id_to_tagsincluster)
        test_id_to_docs = None
        test_ids = set([needs_to_be_merged_test[i][1] for i in range(len(needs_to_be_merged_test))])
        test_ids.add(needs_to_be_merged_test[0][0])
        for id in id_to_tagsincluster:
            if id in test_ids:
                assert test_id_to_docs is None
                test_id_to_docs = id

        assert train_id_to_docs in id_to_tagsincluster and train_id_to_docs in id_to_tagsincluster[train_id_to_docs]
        assert dev_id_to_docs in id_to_tagsincluster and dev_id_to_docs in id_to_tagsincluster[dev_id_to_docs]
        assert test_id_to_docs in id_to_tagsincluster and test_id_to_docs in id_to_tagsincluster[test_id_to_docs]
        all_other_vals = []
        for id, vals in id_to_tagsincluster.items():
            if id != train_id_to_docs and id != dev_id_to_docs and id != test_id_to_docs:
                all_other_vals.append(vals)
        return (train_id_to_docs, id_to_tagsincluster[train_id_to_docs]), \
               (dev_id_to_docs, id_to_tagsincluster[dev_id_to_docs]), \
               (test_id_to_docs, id_to_tagsincluster[test_id_to_docs]), \
               all_other_vals
    else:
        return list(id_to_tagsincluster.values())


def get_list_of_filenames_to_merge_together(df):
    list_of_fnames_to_pair = []
    fname_to_pair_with = None
    for i, row in df.iterrows():
        fname_from_row = row['filename']
        if fname_to_pair_with is None:
            fname_to_pair_with = fname_from_row
        else:
            list_of_fnames_to_pair.append((fname_to_pair_with, fname_from_row))
    return list_of_fnames_to_pair


def main():
    scores, tagpairs = read_in_existing_data('document_bow_cosinesimilarity.txt')
    #make_histogram(scores)
    calculate_number_of_docs_marked_as_duplicates_using_threshold(.9, scores, tagpairs)
    calculate_number_of_docs_marked_as_duplicates_using_threshold(.88, scores, tagpairs)
    calculate_number_of_docs_marked_as_duplicates_using_threshold(.85, scores, tagpairs)


if __name__ == '__main__':
    main()
