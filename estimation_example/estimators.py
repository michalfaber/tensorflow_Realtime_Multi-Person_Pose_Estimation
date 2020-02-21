import math
import cv2
import numpy as np
from config import get_default_configuration


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


if __name__ == '__main__':
    connections = [
        np.array(
            [[ 9. , 14. ,  0.9188747 ,  0.8911759 ,  1.03683093],
            [ 5. , 10. ,  0.8865844 ,  0.8121046 ,  0.94687685],
            [ 8. , 13. ,  0.8726025 ,  0.85033226,  0.9012778 ],
            [ 6. , 11. ,  0.91056985,  0.88929117,  0.88995027],
            [ 7. , 12. ,  0.7325343 ,  0.6240694 ,  0.78442425]]), 
        np.array(
            [[ 9. , 28. ,  0.9188747 ,  0.88108903,  1.00724218],
            [ 8. , 27. ,  0.8726025 ,  0.82121867,  0.96535013],
            [ 5. , 24. ,  0.8865844 ,  0.8704748 ,  0.93374986],
            [ 6. , 25. ,  0.91056985,  0.85670036,  0.92527213],
            [ 7. , 26. ,  0.7325343 ,  0.6815984 ,  0.84348354]]), 
        np.array(
            [[10. , 15. ,  0.8121046 ,  0.91216505,  0.99742639],
            [14. , 19. ,  0.8911759 ,  0.78751093,  0.97054861],
            [13. , 18. ,  0.85033226,  0.617831  ,  0.91969185],
            [11. , 17. ,  0.88929117,  0.7160349 ,  0.83139618],
            [12. , 16. ,  0.6240694 ,  0.39170936,  0.47590637]]), 
        np.array(
            [[15. , 20. ,  0.91216505,  0.6846192 ,  1.09104793],
            [19. , 23. ,  0.78751093,  0.771493  ,  0.89008226],
            [18. , 22. ,  0.617831  ,  0.26212907,  0.63477109],
            [18. , 23. ,  0.617831  ,  0.771493  ,  0.58728189]]), 
        np.array(
            [[28. , 33. ,  0.88108903,  0.72111076,  1.00839311],
            [27. , 32. ,  0.82121867,  0.67423993,  0.98828157],
            [24. , 29. ,  0.8704748 ,  0.81237817,  0.9666315 ],
            [26. , 31. ,  0.6815984 ,  0.4836958 ,  0.80339349],
            [25. , 30. ,  0.85670036,  0.22556348,  0.51399985]]), 
        np.array(
            [[29. , 34. ,  0.81237817,  0.71519095,  0.90577437],
            [33. , 37. ,  0.72111076,  0.5472575 ,  0.65625855],
            [32. , 36. ,  0.67423993,  0.6169104 ,  0.62864926],
            [31. , 35. ,  0.4836958 ,  0.44307223,  0.60174478]]), 
        np.array(
            [[ 9. , 41. ,  0.9188747 ,  0.6915504 ,  0.9004276 ],
            [ 5. , 38. ,  0.8865844 ,  0.64131504,  0.85271515],
            [ 8. , 42. ,  0.8726025 ,  0.6116446 ,  0.83352115],
            [ 6. , 39. ,  0.91056985,  0.62906384,  0.83043894],
            [ 7. , 40. ,  0.7325343 ,  0.3766609 ,  0.62170903]]), 
        np.array(
            [[38. , 44. ,  0.64131504,  0.833872  ,  0.85671546],
            [39. , 43. ,  0.62906384,  0.6413851 ,  0.85363647],
            [42. , 45. ,  0.6116446 ,  0.5693337 ,  0.70891048],
            [40. , 47. ,  0.3766609 ,  0.52756023,  0.66385489],
            [41. , 46. ,  0.6915504 ,  0.34187955,  0.4365968 ]]), 
        np.array(
            [[44. , 48. ,  0.833872  ,  0.646768  ,  0.9471579 ],
            [45. , 50. ,  0.5693337 ,  0.54047376,  0.90375594],
            [43. , 49. ,  0.6413851 ,  0.621414  ,  0.88494388],
            [47. , 52. ,  0.52756023,  0.4758099 ,  0.71417868],
            [46. , 51. ,  0.34187955,  0.29182506,  0.57783459]]), 
        np.array(
            [[ 6. , 54. ,  0.91056985,  0.6280589 ,  0.96156364],
            [ 9. , 56. ,  0.9188747 ,  0.7400102 ,  0.960601  ],
            [ 5. , 53. ,  0.8865844 ,  0.6341558 ,  0.94478426],
            [ 8. , 57. ,  0.8726025 ,  0.46316183,  0.87909239],
            [ 7. , 55. ,  0.7325343 ,  0.3196735 ,  0.60074055]]), 
        np.array(
            [[53. , 59. ,  0.6341558 ,  0.83847153,  0.84733025],
            [54. , 58. ,  0.6280589 ,  0.74924064,  0.81335635],
            [56. , 61. ,  0.7400102 ,  0.85580474,  0.81244228],
            [55. , 62. ,  0.3196735 ,  0.694633  ,  0.5182847 ],
            [57. , 61. ,  0.46316183,  0.85580474,  0.39467257]]), 
        np.array(
            [[59. , 64. ,  0.83847153,  0.72656405,  0.88302198],
            [61. , 66. ,  0.85580474,  0.57901657,  0.82494843],
            [58. , 63. ,  0.74924064,  0.658298  ,  0.82203498],
            [62. , 67. ,  0.694633  ,  0.49369463,  0.74128318],
            [60. , 65. ,  0.41720843,  0.4767233 ,  0.453668  ]]), 
        np.array(
            [[6. , 1. , 0.91056985, 0.9313662 , 0.99411402],
            [8. , 3. , 0.8726025 , 0.9306832 , 0.98572515],
            [9. , 4. , 0.9188747 , 0.9516923 , 0.98134359],
            [5. , 0. , 0.8865844 , 0.92409194, 0.94417737],
            [7. , 2. , 0.7325343 , 0.9052348 , 0.84682636]]), 
        np.array(
            [[ 0. , 68. ,  0.92409194,  0.9424342 ,  1.14795679],
            [ 3. , 71. ,  0.9306832 ,  0.9191763 ,  1.130467  ],
            [ 1. , 69. ,  0.9313662 ,  0.9409762 ,  1.10818533],
            [ 2. , 70. ,  0.9052348 ,  0.8993071 ,  1.10167558],
            [ 4. , 72. ,  0.9516923 ,  0.9394402 ,  1.08395883]]), 
        np.array(
            [[72. , 82. ,  0.9394402 ,  0.82936436,  0.9980747 ],
            [68. , 78. ,  0.9424342 ,  0.92250913,  0.98662932],
            [69. , 79. ,  0.9409762 ,  0.90719926,  0.98011624],
            [70. , 80. ,  0.8993071 ,  0.8572976 ,  0.87976664],
            [71. , 81. ,  0.9191763 ,  0.8653215 ,  0.7991888 ]]), 
        np.array(
            [[ 2. , 75. ,  0.9052348 ,  0.92142344,  0.99353639],
            [ 0. , 73. ,  0.92409194,  0.936771  ,  0.9643986 ],
            [ 3. , 76. ,  0.9306832 ,  0.9070393 ,  0.95751885],
            [ 4. , 77. ,  0.9516923 ,  0.9689289 ,  0.95725774],
            [ 1. , 74. ,  0.9313662 ,  0.93059033,  0.95717775]]), 
        np.array(
            [[77. , 86. ,  0.9689289 ,  0.92744666,  1.03445414],
            [74. , 84. ,  0.93059033,  0.83940786,  0.99919387],
            [76. , 85. ,  0.9070393 ,  0.81429726,  0.95994219],
            [73. , 83. ,  0.936771  ,  0.77801484,  0.87463949]]), 
        np.array(
            [[14. , 82. ,  0.8911759 ,  0.82936436,  0.95386559],
            [10. , 78. ,  0.8121046 ,  0.92250913,  0.911006  ],
            [11. , 79. ,  0.88929117,  0.90719926,  0.89117094],
            [13. , 81. ,  0.85033226,  0.8653215 ,  0.84301486],
            [12. , 80. ,  0.6240694 ,  0.8572976 ,  0.80546115]]), 
        np.array(
            [[28. , 86. ,  0.88108903,  0.92744666,  1.04618902],
            [25. , 84. ,  0.85670036,  0.83940786,  0.98009169],
            [24. , 83. ,  0.8704748 ,  0.77801484,  0.93649059],
            [27. , 85. ,  0.82121867,  0.81429726,  0.929685  ]])]

    cfg = get_default_configuration()
    skeletons = estimate(cfg, connections)

    print(skeletons)
