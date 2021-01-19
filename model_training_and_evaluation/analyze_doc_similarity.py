import matplotlib.pyplot as plt
import os


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


def cluster_tags_above_a_threshold(threshold, sim_scores, corr_tag_pairs):
    # treat anything above the theshold as signifying a link
    tag_to_clusterid = {}
    clusterid_to_alltagsincluster = {}
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


def get_list_of_filenames_to_cluster_together(filename_with_scores='document_bow_cosinesimilarity.txt', threshold=.88):
    assert filename_with_scores is not None and os.path.isfile(filename_with_scores)
    similarity_scores, tag_pairs = read_in_existing_data(filename_with_scores)
    id_to_tagsincluster = cluster_tags_above_a_threshold(threshold, similarity_scores, tag_pairs)
    return list(id_to_tagsincluster.values())


def main():
    scores, tagpairs = read_in_existing_data('document_bow_cosinesimilarity.txt')
    #make_histogram(scores)
    calculate_number_of_docs_marked_as_duplicates_using_threshold(.9, scores, tagpairs)
    calculate_number_of_docs_marked_as_duplicates_using_threshold(.88, scores, tagpairs)
    calculate_number_of_docs_marked_as_duplicates_using_threshold(.85, scores, tagpairs)


if __name__ == '__main__':
    main()
