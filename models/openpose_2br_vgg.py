import tensorflow as tf

from tf_netbuilder.builder import NetModule
from tf_netbuilder.files import download_checkpoint


class OpenPose2BranchesVGG(tf.keras.Model):

    _model_def = {
        'inputs#': [
            ['img#norm05', 'mask_s#', 'mask_l#']
        ],

        # VGG19 backbone

        'backbone#': [
            ['select:img#'],
            ['block1_conv1#cnreg_r1_k3_s1_c64_nre'],
            ['block1_conv2#cnreg_r1_k3_s1_c64_nre'],
            ['maxpool_r1_k2_s2'],
            ['block2_conv1#cnreg_r1_k3_s1_c128_nre'],
            ['block2_conv2#cnreg_r1_k3_s1_c128_nre'],
            ['maxpool_r1_k2_s2'],
            ['block3_conv1#cnreg_r1_k3_s1_c256_nre'],
            ['block3_conv2#cnreg_r1_k3_s1_c256_nre'],
            ['block3_conv3#cnreg_r1_k3_s1_c256_nre'],
            ['block3_conv4#cnreg_r1_k3_s1_c256_nre'],
            ['maxpool_r1_k2_s2'],
            ['block4_conv1#cnreg_r1_k3_s1_c512_nre'],
            ['block4_conv2#cnreg_r1_k3_s1_c512_nre'],
            ['cnreg_r1_k3_s1_c256_nre'],
            ['cnreg_r1_k3_s1_c128_nre']
        ],

        # stage 0 - S (38)

        'stage_0_s#': [
            ['select:backbone#'],
            ['cnreg_r3_k3_s1_c128_nre'],
            ['cnreg_r1_k1_s1_c512_nre'],
            ['cn3_r1_k1_s1_c38']
        ],
        'stage_0_s_masked#': [
            ['select:stage_0_s#:mask_s#'],
            ['mul:'],
        ],

        # stage 0 - L (19)

        'stage_0_l#': [
            ['select:backbone#'],
            ['cnreg_r3_k3_s1_c128_nre'],
            ['cnreg_r1_k1_s1_c512_nre'],
            ['cn3_r1_k1_s1_c19']
        ],
        'stage_0_l_masked#': [
            ['select:stage_0_l#:mask_l#'],
            ['mul:'],
        ],

        # stage 1 - S (38)

        'stage_0_out#': [
            ['select:stage_0_s#:stage_0_l#:backbone#'],
            ['cnct:'],
        ],
        'stage_1_s#': [
            ['select:stage_0_out#'],
            ['cnreg_r5_k7_s1_c128_nre'],
            ['cnreg_r1_k1_s1_c128_nre'],
            ['cn3_r1_k1_s1_c38']
        ],
        'stage_1_s_masked#': [
            ['select:stage_1_s#:mask_s#'],
            ['mul:'],
        ],

        # stage 1 - L (19)

        'stage_1_l#': [
            ['select:stage_0_out#'],
            ['cnreg_r5_k7_s1_c128_nre'],
            ['cnreg_r1_k1_s1_c128_nre'],
            ['cn3_r1_k1_s1_c19']
        ],
        'stage_1_l_masked#': [
            ['select:stage_1_l#:mask_l#'],
            ['mul:'],
        ],

        # stage 2 - S (38)

        'stage_1_out#': [
            ['select:stage_1_s#:stage_1_l#:backbone#'],
            ['cnct:'],
        ],
        'stage_2_s#': [
            ['select:stage_1_out#'],
            ['cnreg_r5_k7_s1_c128_nre'],
            ['cnreg_r1_k1_s1_c128_nre'],
            ['cn3_r1_k1_s1_c38']
        ],
        'stage_2_s_masked#': [
            ['select:stage_2_s#:mask_s#'],
            ['mul:'],
        ],

        # stage 2 - L (19)

        'stage_2_l#': [
            ['select:stage_1_out#'],
            ['cnreg_r5_k7_s1_c128_nre'],
            ['cnreg_r1_k1_s1_c128_nre'],
            ['cn3_r1_k1_s1_c19']
        ],
        'stage_2_l_masked#': [
            ['select:stage_2_l#:mask_l#'],
            ['mul:'],
        ],

        # stage 3 - S (38)

        'stage_2_out#': [
            ['select:stage_2_s#:stage_2_l#:backbone#'],
            ['cnct:'],
        ],
        'stage_3_s#': [
            ['select:stage_2_out#'],
            ['cnreg_r5_k7_s1_c128_nre'],
            ['cnreg_r1_k1_s1_c128_nre'],
            ['cn3_r1_k1_s1_c38']
        ],
        'stage_3_s_masked#': [
            ['select:stage_3_s#:mask_s#'],
            ['mul:'],
        ],

        # stage 3 - L (19)

        'stage_3_l#': [
            ['select:stage_2_out#'],
            ['cnreg_r5_k7_s1_c128_nre'],
            ['cnreg_r1_k1_s1_c128_nre'],
            ['cn3_r1_k1_s1_c19']
        ],
        'stage_3_l_masked#': [
            ['select:stage_3_l#:mask_l#'],
            ['mul:'],
        ],

        # stage 4 - S (38)

        'stage_3_out#': [
            ['select:stage_3_s#:stage_3_l#:backbone#'],
            ['cnct:'],
        ],
        'stage_4_s#': [
            ['select:stage_3_out#'],
            ['cnreg_r5_k7_s1_c128_nre'],
            ['cnreg_r1_k1_s1_c128_nre'],
            ['cn3_r1_k1_s1_c38']
        ],
        'stage_4_s_masked#': [
            ['select:stage_4_s#:mask_s#'],
            ['mul:'],
        ],

        # stage 4 - L (19)

        'stage_4_l#': [
            ['select:stage_3_out#'],
            ['cnreg_r5_k7_s1_c128_nre'],
            ['cnreg_r1_k1_s1_c128_nre'],
            ['cn3_r1_k1_s1_c19']
        ],
        'stage_4_l_masked#': [
            ['select:stage_4_l#:mask_l#'],
            ['mul:'],
        ],

        # stage 5 - S (38)

        'stage_4_out#': [
            ['select:stage_4_s#:stage_4_l#:backbone#'],
            ['cnct:'],
        ],
        'stage_5_s#': [
            ['select:stage_4_out#'],
            ['cnreg_r5_k7_s1_c128_nre'],
            ['cnreg_r1_k1_s1_c128_nre'],
            ['cn3_r1_k1_s1_c38']
        ],
        'stage_5_s_masked#': [
            ['select:stage_5_s#:mask_s#'],
            ['mul:'],
        ],

        # stage 5 - L (19)

        'stage_5_l#': [
            ['select:stage_4_out#'],
            ['cnreg_r5_k7_s1_c128_nre'],
            ['cnreg_r1_k1_s1_c128_nre'],
            ['cn3_r1_k1_s1_c19']
        ],
        'stage_5_l_masked#': [
            ['select:stage_5_l#:mask_l#'],
            ['mul:'],
        ],
    }

    _model_ins = 'inputs#'

    _model_outs = ['stage_0_s_masked#', 'stage_0_l_masked#',
                   'stage_1_s_masked#', 'stage_1_l_masked#',
                   'stage_2_s_masked#', 'stage_2_l_masked#',
                   'stage_3_s_masked#', 'stage_3_l_masked#',
                   'stage_4_s_masked#', 'stage_4_l_masked#',
                   'stage_5_s_masked#', 'stage_5_l_masked#',
                   ]

    def __init__(self, in_chs, training):
        super(OpenPose2BranchesVGG, self).__init__(name="OpenPose2BrVGG")

        model_def = self._model_def
        model_ins = self._model_ins
        model_outs = self._model_outs

        if not training:
            # single input without masks
            model_def['inputs#'] = [
                ['img#norm05']
            ]
            # remove masking - no need in prediction
            model_def = {k: v for k, v in model_def.items() if 'masked' not in k}
            # unmasked outputs
            model_outs = ['stage_0_s#', 'stage_0_l#',
                          'stage_1_s#', 'stage_1_l#',
                          'stage_2_s#', 'stage_2_l#',
                          'stage_3_s#', 'stage_3_l#',
                          'stage_4_s#', 'stage_4_l#',
                          'stage_5_s#', 'stage_5_l#']

        self.net = NetModule(model_def,
                             model_ins,
                             model_outs, in_chs=in_chs, name="VGG")

    def call(self, inputs):
        x = self.net(inputs)

        return x


def create_openpose_2branches_vgg(pretrained=False, training=False):

    pretrained_url = "https://github.com/michalfaber/tensorflow_Realtime_Multi-Person_Pose_Estimation/releases/download/v1.0/openpose_2br_vgg.zip"

    if training:
        model = OpenPose2BranchesVGG(in_chs=[3, 38, 19], training=training)
        model.build([tf.TensorShape((None, None, None, 3)),
                     tf.TensorShape((None, None, None, 38)),
                     tf.TensorShape((None, None, None, 19))
                     ])
    else:
        model = OpenPose2BranchesVGG(in_chs=[3], training=training)
        model.build([tf.TensorShape((None, None, None, 3))])

    if pretrained:
        path = download_checkpoint(pretrained_url)
        model.load_weights(path)

    return model
