import tensorflow as tf

from tf_netbuilder.builder import NetModule
from tf_netbuilder.files import download_checkpoint


class OpenPoseSingleNet(tf.keras.Model):

    _model_def = {
        'inputs#': [
            ['img#norm1']
        ],

        # Mobilenet v3 backbone

        'backbone#': [
            ['select:img#'],
            ['cn_bn_r1_k3_s2_c16_nhs'],
            ['ir_r1_k3_s1_e1_c16_nre'],
            ['ir_r1_k3_s2_e4_c24_nre', 'ir_r1_k3_s1_e3_c24_nre'],
            ['c3#ir_r3_k5_s2_e3_c40_se4_nre'],
            ['ir_r1_k3_s2_e6_c80_nhs'],
            ['ir_r1_k3_s1_e2.5_c80_nhs'],
            ['ir_r2_k3_s1_e2.3_c80_nhs'],
            ['c4#ir_r2_k3_s1_e6_c112_se4_nhs'],
            ['upscaled_c4#up_x2:'],
            ['cnct:c3#:upscaled_c4#']
        ],

        # PAF stages

        'stage_0#': [
            ['select:backbone#'],
            ['cn2_r5_e1_c192_npre'],
            ['ir_r1_k1_s1_e1_c256_se4_npre'],
            ['hd_r1_k1_s1_c38']
        ],
        'stage_1#': [
            ['select:stage_0#:backbone#'],
            ['cnct:'],
            ['cn2_r5_e1_c384_npre'],
            ['ir_r1_k1_s1_e1_c256_se4_npre'],
            ['hd_r1_k1_s1_c38']
        ],
        'stage_2#': [
            ['select:stage_1#:backbone#'],
            ['cnct:'],
            ['cn2_r5_e1_c384_npre'],
            ['ir_r1_k1_s1_e1_c512_se4_npre'],
            ['hd_r1_k1_s1_c38']
        ],

        # Heatmap stages

        'stage_3#': [
            ['select:stage_2#:backbone#'],
            ['cnct:'],
            ['cn2_r5_e1_c384_npre'],
            ['ir_r1_k1_s1_e1_c512_se4_npre'],
            ['hd_r1_k1_s1_c19']
        ],
    }

    _model_ins = 'inputs#'

    _model_outs = ['stage_0#', 'stage_1#', 'stage_2#', 'stage_3#']

    def __init__(self, in_chs):
        super(OpenPoseSingleNet, self).__init__()

        self.net = NetModule(self._model_def,
                             self._model_ins,
                             self._model_outs, in_chs=in_chs, name="MobilenetV3")

    def call(self, inputs):
        x = self.net(inputs)

        return x


def create_openpose_singlenet(pretrained=False):

    pretrained_url = "https://github.com/michalfaber/tensorflow_Realtime_Multi-Person_Pose_Estimation/releases/download/v1.0/openpose_singlenet_v1.zip"

    model = OpenPoseSingleNet(in_chs=[3])
    model.build([tf.TensorShape((None, 224, 224, 3))])

    if pretrained:
        path = download_checkpoint(pretrained_url)
        model.load_weights(path)

    return model
