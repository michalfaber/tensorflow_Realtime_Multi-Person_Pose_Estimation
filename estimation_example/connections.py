import math
import numpy as np
from config import get_default_configuration


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


if __name__ == '__main__':

    coords = {'nose': 
        [(173, 13, 0.92409194, 0), 
        (85, 23, 0.9313662, 1), 
        (135, 29, 0.9052348, 2), 
        (19, 79, 0.9306832, 3), 
        (48, 83, 0.9516923, 4)], 
        'neck': 
        [(172, 24, 0.8865844, 5), 
        (85, 33, 0.91056985, 6), 
        (129, 42, 0.7325343, 7), 
        (18, 89, 0.8726025, 8), 
        (47, 90, 0.9188747, 9)], 
        'right_shoulder': 
        [(164, 26, 0.8121046, 10), 
        (76, 34, 0.88929117, 11), 
        (117, 42, 0.6240694, 12), 
        (11, 89, 0.85033226, 13), 
        (39, 90, 0.8911759, 14)], 
        'right_elbow': 
        [(153, 39, 0.91216505, 15), 
        (97, 42, 0.39170936, 16), 
        (73, 49, 0.7160349, 17), 
        (18, 101, 0.617831, 18), 
        (38, 106, 0.78751093, 19)], 
        'right_wrist': 
        [(160, 51, 0.6846192, 20), 
        (80, 52, 0.65054333, 21), 
        (24, 103, 0.26212907, 22), 
        (47, 111, 0.771493, 23)], 
        'left_shoulder': 
        [(181, 22, 0.8704748, 24), 
        (95, 32, 0.85670036, 25), 
        (141, 41, 0.6815984, 26), 
        (26, 89, 0.82121867, 27), 
        (56, 90, 0.88108903, 28)], 
        'left_elbow': 
        [(185, 37, 0.81237817, 29), 
        (99, 44, 0.22556348, 30), 
        (152, 56, 0.4836958, 31), 
        (34, 98, 0.67423993, 32), 
        (57, 105, 0.72111076, 33)], 
        'left_wrist': 
        [(185, 49, 0.71519095, 34), 
        (158, 57, 0.44307223, 35), 
        (27, 103, 0.6169104, 36), 
        (52, 111, 0.5472575, 37)], 
        'right_hip': 
        [(172, 59, 0.64131504, 38), 
        (82, 63, 0.62906384, 39), 
        (122, 96, 0.3766609, 40), 
        (43, 112, 0.6915504, 41), 
        (10, 115, 0.6116446, 42)], 
        'right_knee': 
        [(87, 83, 0.6413851, 43), 
        (174, 83, 0.833872, 44), 
        (22, 101, 0.5693337, 45), 
        (37, 110, 0.34187955, 46), 
        (142, 128, 0.52756023, 47)], 
        'right_ankle': 
        [(173, 106, 0.646768, 48), 
        (90, 108, 0.621414, 49), 
        (32, 119, 0.54047376, 50), 
        (53, 119, 0.29182506, 51), 
        (149, 167, 0.4758099, 52)], 
        'left_hip': 
        [(182, 57, 0.6341558, 53), 
        (95, 62, 0.6280589, 54), 
        (129, 94, 0.3196735, 55), 
        (53, 111, 0.7400102, 56), 
        (22, 114, 0.46316183, 57)], 
        'left_knee': 
        [(96, 82, 0.74924064, 58), 
        (182, 82, 0.83847153, 59), 
        (31, 100, 0.41720843, 60), 
        (67, 108, 0.85580474, 61), 
        (130, 136, 0.694633, 62)], 
        'left_ankle': 
        [(83, 102, 0.658298, 63), 
        (179, 104, 0.72656405, 64), 
        (39, 120, 0.4767233, 65), 
        (46, 121, 0.57901657, 66), 
        (115, 166, 0.49369463, 67)], 
        'right_eye': 
        [(170, 11, 0.9424342, 68), 
        (83, 21, 0.9409762, 69), 
        (131, 25, 0.8993071, 70), 
        (17, 77, 0.9191763, 71), 
        (46, 81, 0.9394402, 72)], 
        'left_eye': 
        [(174, 11, 0.936771, 73), 
        (86, 21, 0.93059033, 74), 
        (137, 25, 0.92142344, 75), 
        (20, 78, 0.9070393, 76), 
        (50, 81, 0.9689289, 77)], 
        'right_ear': 
        [(167, 13, 0.92250913, 78), 
        (80, 23, 0.90719926, 79), 
        (123, 27, 0.8572976, 80), 
        (13, 79, 0.8653215, 81), 
        (44, 81, 0.82936436, 82)], 
        'left_ear': 
        [(177, 11, 0.77801484, 83), 
        (89, 22, 0.83940786, 84), 
        (23, 79, 0.81429726, 85), 
        (53, 82, 0.92744666, 86)]}

    paf_path = './resources/pafs.npy'
    paf = np.load(paf_path)

    cfg = get_default_configuration()
    connections = get_connections(cfg, coords, paf)

    print(connections)
