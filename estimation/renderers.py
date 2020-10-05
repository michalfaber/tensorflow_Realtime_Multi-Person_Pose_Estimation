import cv2
import math
import numpy as np


def draw(config, input_image, coords, subset, resize_fac = 1):

    stickwidth = 1

    canvas = input_image.copy()

    for body_part_type, body_part_meta in config.body_parts.items():
        color = body_part_meta.color
        body_part_peaks = coords[body_part_type.name]

        for peak in body_part_peaks:
            a = peak[0] * resize_fac
            b = peak[1] * resize_fac
            cv2.circle(canvas, (a, b), stickwidth, color, thickness=-1)

    # dict(id: [y,x]) Note, coord are reversed
    xy_by_id = dict([(item[3], np.array([item[1], item[0]])) for sublist in coords.values() for item in sublist])

    xy = np.zeros((2,2))
    for i, conn_type in enumerate(config.connection_types):
        index1 = config.body_parts[conn_type.from_body_part].slot_idx
        index2 = config.body_parts[conn_type.to_body_part].slot_idx
        indexes = np.array([index1, index2])        
        for s in subset:

            ids = s[indexes]            
            if -1 in ids:
                continue

            cur_canvas = canvas.copy()
            xy[0, :] = xy_by_id[ids[0]]
            xy[1, :] = xy_by_id[ids[1]]
            
            m_x = np.mean(xy[:, 0])
            m_y = np.mean(xy[:, 1])
            sss = xy[1, 1]
            length = ((xy[0, 0] - xy[1, 0]) ** 2 + (xy[0, 1] - xy[1, 1]) ** 2) ** 0.5

            angle = math.degrees(math.atan2(xy[0, 0] - xy[1, 0], xy[0, 1] - xy[1, 1]))

            polygon = cv2.ellipse2Poly((int(m_y * resize_fac), int(m_x * resize_fac)),
                                       (int(length * resize_fac / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, conn_type.color)
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas
