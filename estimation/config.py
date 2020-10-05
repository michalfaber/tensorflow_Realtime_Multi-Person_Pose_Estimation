from enum import IntEnum


class BodyPart(IntEnum):
    """
    List of all body parts
    """
    nose = 0
    neck = 1  # this part is not from COCO
    right_shoulder = 2
    right_elbow = 3
    right_wrist = 4
    left_shoulder = 5
    left_elbow = 6
    left_wrist = 7
    right_hip = 8
    right_knee = 9
    right_ankle = 10
    left_hip = 11
    left_knee = 12
    left_ankle = 13
    right_eye = 14
    left_eye = 15
    right_ear = 16
    left_ear = 17
    background = 18


class ConnectionMeta:
    """
    Metadata for each connection type:

    -first body part identifier, connections are defined beteen 2 body parts
    -second body part identifier
    -index in paf (dx) 
    -index in paf (dy) 
    -color, helpful for rendering this connection
    """
    def __init__(self, from_body_part: BodyPart, to_body_part: BodyPart, paf_dx_idx: int, 
                paf_dy_idx: int, color: list):
        self.from_body_part = from_body_part
        self.to_body_part = to_body_part
        self.paf_dx_idx = paf_dx_idx
        self.paf_dy_idx = paf_dy_idx
        self.color = color


class BodyPartMeta:
    """
    Metadata for each body part type:

    -body part identifier
    -index in heatmap where the relevant peaks can be found for this body part type
    -slot index for this body part. During the estimation phase, each skeleton has an array containing 
    identifiers of body parts which belong to this skeleton. Each such identifier has to be stored at an 
    specific position in the array. This position is being kept here as a slot_idx
    -color, helpful for rendering this body part
    """
    def __init__(self, body_part: BodyPart, heatmap_idx: int, slot_idx: int, color: list):
        self.body_part = body_part
        self.heatmap_idx = heatmap_idx
        self.slot_idx = slot_idx
        self.color = color


class ConnectionsConfig:
    """
    Configuration of all body part types and connection types beetween them. This architecture allows you
    to register only a subset of body parts and connections. Less connections, faster estimation.
    """
    body_parts = dict()
    connection_types = []
    
    def __init__(self):
        self.slot_idx_seq = 0

    def register_body_part(self, body_part: BodyPart, heatmap_idx: int, color: list):
        """
        Registers a body part
        """
        self.body_parts[body_part] = BodyPartMeta(body_part, heatmap_idx, self.slot_idx_seq, color)
        self.slot_idx_seq += 1

    def add_connection(self, from_body_part: BodyPart, to_body_part: BodyPart, paf_dx_idx: int, paf_dy_idx: int, color: list):
        """
        Adds a connection definition between two body parts. An Exception will be raise if the body part is not registered
        """
        if from_body_part not in self.body_parts.keys():
            raise Exception(f"Body part '{from_body_part.name}' is not registered.")
        if to_body_part not in self.body_parts.keys():
            raise Exception(f"Body part '{to_body_part.name}' is not registered.")
        self.connection_types.append(ConnectionMeta(from_body_part, to_body_part, paf_dx_idx, paf_dy_idx, color))

    def conn_types_size(self):
        """
        Returns the number of all connection types
        """
        return len(self.connection_types)

    def body_parts_size(self):
        """
        Returns the number of all registered body parts
        """
        return len(self.body_parts)        


def get_default_configuration():
    """
    This is the default configuration including all body parts and connections. 
    You may remove the last 2 connections - ears to shoulders. Why did the CMU include them in their solution ?
    """
    config = ConnectionsConfig()
    config.register_body_part(body_part = BodyPart.nose, heatmap_idx = 0, color = [255, 0, 0])
    config.register_body_part(body_part = BodyPart.neck, heatmap_idx = 1, color = [255, 85, 0])
    config.register_body_part(body_part = BodyPart.right_shoulder, heatmap_idx = 2, color = [255, 170, 0])
    config.register_body_part(body_part = BodyPart.right_elbow, heatmap_idx = 3, color = [255, 255, 0])
    config.register_body_part(body_part = BodyPart.right_wrist, heatmap_idx = 4, color = [170, 255, 0])
    config.register_body_part(body_part = BodyPart.left_shoulder, heatmap_idx = 5, color = [85, 255, 0])
    config.register_body_part(body_part = BodyPart.left_elbow, heatmap_idx = 6, color = [0, 255, 0])
    config.register_body_part(body_part = BodyPart.left_wrist, heatmap_idx = 7, color = [0, 255, 85])
    config.register_body_part(body_part = BodyPart.right_hip, heatmap_idx = 8, color = [0, 255, 170])
    config.register_body_part(body_part = BodyPart.right_knee, heatmap_idx = 9, color = [0, 255, 255])
    config.register_body_part(body_part = BodyPart.right_ankle, heatmap_idx = 10, color = [0, 170, 255])
    config.register_body_part(body_part = BodyPart.left_hip, heatmap_idx = 11, color = [0, 85, 255])
    config.register_body_part(body_part = BodyPart.left_knee, heatmap_idx = 12, color = [0, 0, 255])
    config.register_body_part(body_part = BodyPart.left_ankle, heatmap_idx = 13, color = [170, 0, 255])
    config.register_body_part(body_part = BodyPart.right_eye, heatmap_idx = 14, color = [255, 0, 255])
    config.register_body_part(body_part = BodyPart.left_eye, heatmap_idx = 15, color = [255, 0, 170])
    config.register_body_part(body_part = BodyPart.right_ear, heatmap_idx = 16, color = [255, 0, 85])
    config.register_body_part(body_part = BodyPart.left_ear, heatmap_idx = 17, color = [255, 0, 85])

    config.add_connection(from_body_part = BodyPart.neck,           to_body_part = BodyPart.right_shoulder, paf_dx_idx = 12, paf_dy_idx = 13, color = [255, 0, 0])
    config.add_connection(from_body_part = BodyPart.neck,           to_body_part = BodyPart.left_shoulder,  paf_dx_idx = 20, paf_dy_idx = 21, color = [255, 85, 0])
    config.add_connection(from_body_part = BodyPart.right_shoulder, to_body_part = BodyPart.right_elbow,    paf_dx_idx = 14, paf_dy_idx = 15, color = [255, 170, 0])
    config.add_connection(from_body_part = BodyPart.right_elbow,    to_body_part = BodyPart.right_wrist,    paf_dx_idx = 16, paf_dy_idx = 17, color = [255, 255, 0])
    config.add_connection(from_body_part = BodyPart.left_shoulder,  to_body_part = BodyPart.left_elbow,     paf_dx_idx = 22, paf_dy_idx = 23, color = [170, 255, 0])
    config.add_connection(from_body_part = BodyPart.left_elbow,     to_body_part = BodyPart.left_wrist,     paf_dx_idx = 24, paf_dy_idx = 25, color = [85, 255, 0])
    config.add_connection(from_body_part = BodyPart.neck,           to_body_part = BodyPart.right_hip,      paf_dx_idx = 0,  paf_dy_idx = 1,  color = [0, 255, 0])
    config.add_connection(from_body_part = BodyPart.right_hip,      to_body_part = BodyPart.right_knee,     paf_dx_idx = 2,  paf_dy_idx = 3,  color = [0, 255, 85])
    config.add_connection(from_body_part = BodyPart.right_knee,     to_body_part = BodyPart.right_ankle,    paf_dx_idx = 4,  paf_dy_idx = 5,  color = [0, 255, 170])
    config.add_connection(from_body_part = BodyPart.neck,           to_body_part = BodyPart.left_hip,       paf_dx_idx = 6,  paf_dy_idx = 7,  color = [0, 255, 255])
    config.add_connection(from_body_part = BodyPart.left_hip,       to_body_part = BodyPart.left_knee,      paf_dx_idx = 8,  paf_dy_idx = 9,  color = [0, 170, 255])
    config.add_connection(from_body_part = BodyPart.left_knee,      to_body_part = BodyPart.left_ankle,     paf_dx_idx = 10, paf_dy_idx = 11, color = [0, 85, 255])
    config.add_connection(from_body_part = BodyPart.neck,           to_body_part = BodyPart.nose,           paf_dx_idx = 28, paf_dy_idx = 29, color = [0, 0, 255])
    config.add_connection(from_body_part = BodyPart.nose,           to_body_part = BodyPart.right_eye,      paf_dx_idx = 30, paf_dy_idx = 31, color = [85, 0, 255])
    config.add_connection(from_body_part = BodyPart.right_eye,      to_body_part = BodyPart.right_ear,      paf_dx_idx = 34, paf_dy_idx = 35, color = [170, 0, 255])
    config.add_connection(from_body_part = BodyPart.nose,           to_body_part = BodyPart.left_eye,       paf_dx_idx = 32, paf_dy_idx = 33, color = [255, 0, 255])
    config.add_connection(from_body_part = BodyPart.left_eye,       to_body_part = BodyPart.left_ear,       paf_dx_idx = 36, paf_dy_idx = 37, color = [255, 0, 170])
    config.add_connection(from_body_part = BodyPart.right_shoulder, to_body_part = BodyPart.right_ear,      paf_dx_idx = 18, paf_dy_idx = 19, color = [255, 0, 85])
    config.add_connection(from_body_part = BodyPart.left_shoulder,  to_body_part = BodyPart.left_ear,       paf_dx_idx = 26, paf_dy_idx = 27, color = [255, 0, 85])
    
    return config
