import math
import numpy as np


def get_connections(
        config, coords, paf, threshold=0.05, mid_num=10, minimum_mid_num=8):
    """
    Finds the connection candidates and returns only valid connections.

    :param config: pose estimation configuration.
    :param coords: dictionary with coordinates of all body parts.
    :param paf: paf maps.
    :param threshold: threshold for the intensity value in paf for a given mid point. If value at a mid point
    is below the threshold the mid point is not taken into account.
    :param mid_num: number of mid point for sampling
    :param minimum_mid_num: minimum number of valid mid points for the connection candidate
    :return: list of arrays containing identified connections of a given type :
        [
            array(
              [id1, id2, score1, score2, total_score]
              [id1, id2, score1, score2, total_score]
              ...
            ),
            array(
            ...
            )
        ]
    """
    all_cand_connections = []

    for conn in config.connection_types:

        # select dx and dy PAFs for this connection type
        paf_dx = paf[:, :, conn.paf_dx_idx]
        paf_dy = paf[:, :, conn.paf_dy_idx]

        # get coordinates lists for 2 body part types which belong to the current connection type
        cand_a = coords[conn.from_body_part.name]
        cand_b = coords[conn.to_body_part.name]

        n_a = len(cand_a)
        n_b = len(cand_b)
        max_connections = min(n_a, n_b)

        # lets check each combination of detected 2 body parts - candidate connections
        if n_a != 0 and n_b != 0:

            # here we will store the connection candidates 5 columns:
            # [ body part id1, body part id2, body part score1, body part score2, total score of connection ]
            connection_candidates = np.zeros((0, 5))

            for i in range(n_a):
                for j in range(n_b):
                    # find the distance between the 2 body parts. The expression cand_b[j][:2]
                    # returns an 2 element array with coordinates x,y
                    vec = np.subtract(cand_b[j][:2], cand_a[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])

                    # skip the connection if 2 body parts overlaps
                    if norm == 0:
                        continue

                    # normalize the vector
                    vec = np.divide(vec, norm)

                    # get the set midpoints between 2 body parts (their coordinates x,y)
                    start_end = list(zip(np.linspace(cand_a[i][0], cand_b[j][0], num=mid_num),
                                         np.linspace(cand_a[i][1], cand_b[j][1], num=mid_num)))

                    # having the coord of midpoint we can read the intensity value in paf map at the midpoint
                    # for dx component
                    vec_x = np.array(
                        [paf_dx[int(round(start_end[i][1])), int(
                            round(start_end[i][0]))] for i in range(mid_num)]
                    )
                    # for dy component
                    vec_y = np.array(
                        [paf_dy[int(round(start_end[i][1])), int(
                            round(start_end[i][0]))] for i in range(mid_num)]
                    )

                    # calculate the score for the connection weighted by the distance between body parts
                    score_midpts = np.multiply(
                        vec_x, vec[0]) + np.multiply(vec_y, vec[1])

                    # get the total score
                    total_score = sum(score_midpts) / len(score_midpts)

                    # number of midpoints with intensity above the threshold shouldn't be less than 80% of all midpoints
                    criterion1 = len(np.nonzero(
                        score_midpts > threshold)[0]) > minimum_mid_num
                    criterion2 = total_score > 0

                    if criterion1 and criterion2:
                        # add this connection to the list  [id1, id2, score1, score2, total score]
                        connection_candidates = np.vstack(
                            [connection_candidates,
                             [cand_a[i][3],
                              cand_b[j][3],
                              cand_a[i][2],
                              cand_b[j][2],
                              total_score]])

            # sort the array by the total score - descending. (the sorted array is reversed by the expression [::-1])
            sorted_connections = connection_candidates[
                connection_candidates[:, 4].argsort()][::-1]

            # make sure we get no more than max_connections
            all_cand_connections.append(
                sorted_connections[:max_connections, :])
        else:
            # not found any body parts but we still need to add empty list to preserve the correct indexing in the
            # output array
            all_cand_connections.append([])

    return all_cand_connections
