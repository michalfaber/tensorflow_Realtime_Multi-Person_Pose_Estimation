import numpy as np


def estimate(config, connections, min_num_body_parts=4, min_score=0.4):
    """
    Estimates the skeletons.

    :param config: pose estimation configuration.
    :param connections: valid connections
    :param min_num_body_parts: minimum number of body parts for a skeleton
    :param min_score: minimum score value for the skeleton
    :return: list of skeletons. Each skeleton has a list of identifiers of body parts:
        [
            [id1, id2,...,idN, parts_num, score],
            [id1, id2,...,idN, parts_num, score]
            ...
        ]
    """

    # 2 extra slots for number of valid parts and overall score
    number_of_slots = config.body_parts_size() + 2

    # the connections are solely used to group body parts into separate skeletons. As a result we will
    # get an array where each row represents a skeleton, each column contains an identifier of
    # specific body part belonging to a skeleton (plus 2 extra columns for: num of parts and skeleton total score)
    # we will be adding the skeletons to this array:
    subset = np.empty((0, number_of_slots))

    for k, conn in enumerate(config.connection_types):
        if len(connections[k]) > 0:
            # retrieve id and score of all body parts pairs for the current connection type
            part_a = connections[k][:, [0, 2]]  # idA, scoreA
            part_b = connections[k][:, [1, 2]]  # idB, scoreB

            # determine the slot number for 2 body parts types
            slot_idx_a = config.body_parts[conn.from_body_part].slot_idx
            slot_idx_b = config.body_parts[conn.to_body_part].slot_idx

            # iterate over all connection candidates filling up the subset with the correct body part identifiers
            for i in range(len(connections[k])):
                found = 0
                slot_idx = [-1, -1]
                for j in range(len(subset)):
                    if subset[j][slot_idx_a] == part_a[i, 0] or \
                            subset[j][slot_idx_b] == part_b[i, 0]:
                        slot_idx[found] = j
                        found += 1

                if found == 1:
                    j = slot_idx[0]
                    if subset[j][slot_idx_b] != part_b[i, 0]:
                        subset[j][slot_idx_b] = part_b[i, 0]
                        subset[j][-1] += 1
                        subset[j][-2] += part_b[i, 1] + connections[k][i][2]

                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = slot_idx
                    membership = ((subset[j1] >= 0).astype(int) +
                                  (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connections[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][slot_idx_b] = part_b[i, 0]
                        subset[j1][-1] += 1
                        subset[j1][-2] += part_b[i, 1] + connections[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found:
                    row = -1 * np.ones(number_of_slots)
                    row[slot_idx_a] = part_a[i, 0]
                    row[slot_idx_b] = part_b[i, 0]
                    row[-1] = 2
                    row[-2] = part_a[i, 1] + part_b[i, 1] + connections[k][i][2]
                    subset = np.vstack([subset, row])

    # delete some rows of subset which has few parts occur
    delete_idx = []
    for i in range(len(subset)):
        if subset[i][-1] < min_num_body_parts or \
                subset[i][-2] / subset[i][-1] < min_score:
            delete_idx.append(i)
    subset = np.delete(subset, delete_idx, axis=0)

    return subset
