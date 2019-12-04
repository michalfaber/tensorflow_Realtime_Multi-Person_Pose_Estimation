import tensorflow as tf


def conv2d(filters, kernel_size, prefix, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-4),
           bias_regularizer=tf.keras.regularizers.l2(0.0)):
    return tf.keras.layers.Conv2D(filters, kernel_size, activation=activation,
                                  padding='same',
                                  kernel_initializer=tf.initializers.RandomNormal(stddev=0.01),
                                  bias_initializer='zeros',
                                  kernel_regularizer=kernel_regularizer,
                                  bias_regularizer=bias_regularizer,
                                  name=prefix)


class CmuModel(tf.keras.Model):

    def __init__(self, masked_outputs=True):
        super(CmuModel, self).__init__()

        self.masked_outputs = masked_outputs

        self.img_norm = tf.keras.layers.Lambda(lambda x: x / 256 - 0.5)

        # vgg layers

        self.conv1 = conv2d(64, 3, 'conv1_1')
        self.conv2 = conv2d(64, 3, 'conv1_2')
        self.max_pool1 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool1_1')
        self.conv3 = conv2d(128, 3, 'conv2_1')
        self.conv4 = conv2d(128, 3, 'conv2_2')
        self.max_pool2 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool2_1')
        self.conv5 = conv2d(256, 3, 'conv3_1')
        self.conv6 = conv2d(256, 3, 'conv3_2')
        self.conv7 = conv2d(256, 3, 'conv3_3')
        self.conv8 = conv2d(256, 3, 'conv3_4')
        self.max_pool3 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool3_1')
        self.conv9 = conv2d(512, 3, 'conv4_1')
        self.conv10 = conv2d(512, 3, 'conv4_2')  # VGG transfer learning ends here
        self.conv_cpm_1 = conv2d(256, 3, 'conv4_3_CPM')
        self.conv_cpm_2 = conv2d(128, 3, 'conv_4_4_CPM')

        # stage 1

        self.stage1_conv1_s = conv2d(128, 3, "conv_stage1_1")
        self.stage1_conv2_s = conv2d(128, 3, "conv_stage1_2")
        self.stage1_conv3_s = conv2d(128, 3, "conv_stage1_3")
        self.stage1_conv4_s = conv2d(512, 1, "conv_stage1_4")
        self.stage1_conv5_s = conv2d(38, 1, "conv_stage1_5", activation=None)
        self.stage1_conv1_l = conv2d(128, 3, "conv_stage1_6")
        self.stage1_conv2_l = conv2d(128, 3, "conv_stage1_7")
        self.stage1_conv3_l = conv2d(128, 3, "conv_stage1_8")
        self.stage1_conv4_l = conv2d(512, 1, "conv_stage1_9")
        self.stage1_conv5_l = conv2d(19, 1, "conv_stage1_10", activation=None)
        self.stage1_concat = tf.keras.layers.Concatenate(axis=3)

        # stage 2

        self.stage2_conv1_s = conv2d(128, 7, "conv_stage2_1")
        self.stage2_conv2_s = conv2d(128, 7, "conv_stage2_2")
        self.stage2_conv3_s = conv2d(128, 7, "conv_stage2_3")
        self.stage2_conv4_s = conv2d(128, 7, "conv_stage2_4")
        self.stage2_conv5_s = conv2d(128, 7, "conv_stage2_5")
        self.stage2_conv6_s = conv2d(128, 1, "conv_stage2_6")
        self.stage2_conv7_s = conv2d(38, 1, "conv_stage2_7", activation=None)
        self.stage2_conv1_l = conv2d(128, 7, "conv_stage2_8")
        self.stage2_conv2_l = conv2d(128, 7, "conv_stage2_9")
        self.stage2_conv3_l = conv2d(128, 7, "conv_stage2_10")
        self.stage2_conv4_l = conv2d(128, 7, "conv_stage2_11")
        self.stage2_conv5_l = conv2d(128, 7, "conv_stage2_12")
        self.stage2_conv6_l = conv2d(128, 1, "conv_stage2_13")
        self.stage2_conv7_l = conv2d(19, 1, "conv_stage2_14", activation=None)
        self.stage2_concat = tf.keras.layers.Concatenate(axis=3)

        # stage 3

        self.stage3_conv1_s = conv2d(128, 7, "conv_stage3_1")
        self.stage3_conv2_s = conv2d(128, 7, "conv_stage3_2")
        self.stage3_conv3_s = conv2d(128, 7, "conv_stage3_3")
        self.stage3_conv4_s = conv2d(128, 7, "conv_stage3_4")
        self.stage3_conv5_s = conv2d(128, 7, "conv_stage3_5")
        self.stage3_conv6_s = conv2d(128, 1, "conv_stage3_6")
        self.stage3_conv7_s = conv2d(38, 1, "conv_stage3_7", activation=None)
        self.stage3_conv1_l = conv2d(128, 7, "conv_stage3_8")
        self.stage3_conv2_l = conv2d(128, 7, "conv_stage3_9")
        self.stage3_conv3_l = conv2d(128, 7, "conv_stage3_10")
        self.stage3_conv4_l = conv2d(128, 7, "conv_stage3_11")
        self.stage3_conv5_l = conv2d(128, 7, "conv_stage3_12")
        self.stage3_conv6_l = conv2d(128, 1, "conv_stage3_13")
        self.stage3_conv7_l = conv2d(19, 1, "conv_stage3_14", activation=None)
        self.stage3_concat = tf.keras.layers.Concatenate(axis=3)

        # stage 4

        self.stage4_conv1_s = conv2d(128, 7, "conv_stage4_1")
        self.stage4_conv2_s = conv2d(128, 7, "conv_stage4_2")
        self.stage4_conv3_s = conv2d(128, 7, "conv_stage4_3")
        self.stage4_conv4_s = conv2d(128, 7, "conv_stage4_4")
        self.stage4_conv5_s = conv2d(128, 7, "conv_stage4_5")
        self.stage4_conv6_s = conv2d(128, 1, "conv_stage4_6")
        self.stage4_conv7_s = conv2d(38, 1, "conv_stage4_7", activation=None)
        self.stage4_conv1_l = conv2d(128, 7, "conv_stage4_8")
        self.stage4_conv2_l = conv2d(128, 7, "conv_stage4_9")
        self.stage4_conv3_l = conv2d(128, 7, "conv_stage4_10")
        self.stage4_conv4_l = conv2d(128, 7, "conv_stage4_11")
        self.stage4_conv5_l = conv2d(128, 7, "conv_stage4_12")
        self.stage4_conv6_l = conv2d(128, 1, "conv_stage4_13")
        self.stage4_conv7_l = conv2d(19, 1, "conv_stage4_14", activation=None)
        self.stage4_concat = tf.keras.layers.Concatenate(axis=3)

        # stage 5

        self.stage5_conv1_s = conv2d(128, 7, "conv_stage5_1")
        self.stage5_conv2_s = conv2d(128, 7, "conv_stage5_2")
        self.stage5_conv3_s = conv2d(128, 7, "conv_stage5_3")
        self.stage5_conv4_s = conv2d(128, 7, "conv_stage5_4")
        self.stage5_conv5_s = conv2d(128, 7, "conv_stage5_5")
        self.stage5_conv6_s = conv2d(128, 1, "conv_stage5_6")
        self.stage5_conv7_s = conv2d(38, 1, "conv_stage5_7", activation=None)
        self.stage5_conv1_l = conv2d(128, 7, "conv_stage5_8")
        self.stage5_conv2_l = conv2d(128, 7, "conv_stage5_9")
        self.stage5_conv3_l = conv2d(128, 7, "conv_stage5_10")
        self.stage5_conv4_l = conv2d(128, 7, "conv_stage5_11")
        self.stage5_conv5_l = conv2d(128, 7, "conv_stage5_12")
        self.stage5_conv6_l = conv2d(128, 1, "conv_stage5_13")
        self.stage5_conv7_l = conv2d(19, 1, "conv_stage5_14", activation=None)
        self.stage5_concat = tf.keras.layers.Concatenate(axis=3)

        # stage 6

        self.stage6_conv1_s = conv2d(128, 7, "conv_stage6_1")
        self.stage6_conv2_s = conv2d(128, 7, "conv_stage6_2")
        self.stage6_conv3_s = conv2d(128, 7, "conv_stage6_3")
        self.stage6_conv4_s = conv2d(128, 7, "conv_stage6_4")
        self.stage6_conv5_s = conv2d(128, 7, "conv_stage6_5")
        self.stage6_conv6_s = conv2d(128, 1, "conv_stage6_6")
        self.stage6_conv7_s = conv2d(38, 1, "conv_stage6_7", activation=None)
        self.stage6_conv1_l = conv2d(128, 7, "conv_stage6_8")
        self.stage6_conv2_l = conv2d(128, 7, "conv_stage6_9")
        self.stage6_conv3_l = conv2d(128, 7, "conv_stage6_10")
        self.stage6_conv4_l = conv2d(128, 7, "conv_stage6_11")
        self.stage6_conv5_l = conv2d(128, 7, "conv_stage6_12")
        self.stage6_conv6_l = conv2d(128, 1, "conv_stage6_13")
        self.stage6_conv7_l = conv2d(19, 1, "conv_stage6_14", activation=None)

        if self.masked_outputs:
            self.masked_1_s = tf.keras.layers.Multiply()
            self.masked_1_l = tf.keras.layers.Multiply()
            self.masked_2_s = tf.keras.layers.Multiply()
            self.masked_2_l = tf.keras.layers.Multiply()
            self.masked_3_s = tf.keras.layers.Multiply()
            self.masked_3_l = tf.keras.layers.Multiply()
            self.masked_4_s = tf.keras.layers.Multiply()
            self.masked_4_l = tf.keras.layers.Multiply()
            self.masked_5_s = tf.keras.layers.Multiply()
            self.masked_5_l = tf.keras.layers.Multiply()
            self.masked_6_s = tf.keras.layers.Multiply()
            self.masked_6_l = tf.keras.layers.Multiply()

    def call(self, inputs):

        inputs_with_masks = isinstance(inputs, (list, tuple)) and len(inputs) == 3

        img = inputs[0] if inputs_with_masks else inputs

        x = self.img_norm(img)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.max_pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.max_pool3(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv_cpm_1(x)
        vgg = self.conv_cpm_2(x)

        # stage 1

        x = self.stage1_conv1_s(vgg)
        x = self.stage1_conv2_s(x)
        x = self.stage1_conv3_s(x)
        x = self.stage1_conv4_s(x)
        s_1 = self.stage1_conv5_s(x)
        x = self.stage1_conv1_l(vgg)
        x = self.stage1_conv2_l(x)
        x = self.stage1_conv3_l(x)
        x = self.stage1_conv4_l(x)
        l_1 = self.stage1_conv5_l(x)
        stage1_out = self.stage1_concat([s_1, l_1, vgg])

        if self.masked_outputs and inputs_with_masks:
            output_1_s = self.masked_1_s([s_1, inputs[1]])
            output_1_l = self.masked_1_l([l_1, inputs[2]])

        # stage 2

        x = self.stage2_conv1_s(stage1_out)
        x = self.stage2_conv2_s(x)
        x = self.stage2_conv3_s(x)
        x = self.stage2_conv4_s(x)
        x = self.stage2_conv5_s(x)
        x = self.stage2_conv6_s(x)
        s_2 = self.stage2_conv7_s(x)
        x = self.stage2_conv1_l(stage1_out)
        x = self.stage2_conv2_l(x)
        x = self.stage2_conv3_l(x)
        x = self.stage2_conv4_l(x)
        x = self.stage2_conv5_l(x)
        x = self.stage2_conv6_l(x)
        l_2 = self.stage2_conv7_l(x)
        stage2_out = self.stage2_concat([s_2, l_2, vgg])

        if self.masked_outputs and inputs_with_masks:
            output_2_s = self.masked_2_s([s_2, inputs[1]])
            output_2_l = self.masked_2_l([l_2, inputs[2]])

        # stage 3

        x = self.stage3_conv1_s(stage2_out)
        x = self.stage3_conv2_s(x)
        x = self.stage3_conv3_s(x)
        x = self.stage3_conv4_s(x)
        x = self.stage3_conv5_s(x)
        x = self.stage3_conv6_s(x)
        s_3 = self.stage3_conv7_s(x)
        x = self.stage3_conv1_l(stage2_out)
        x = self.stage3_conv2_l(x)
        x = self.stage3_conv3_l(x)
        x = self.stage3_conv4_l(x)
        x = self.stage3_conv5_l(x)
        x = self.stage3_conv6_l(x)
        l_3 = self.stage3_conv7_l(x)
        stage3_out = self.stage3_concat([s_3, l_3, vgg])

        if self.masked_outputs and inputs_with_masks:
            output_3_s = self.masked_3_s([s_3, inputs[1]])
            output_3_l = self.masked_3_l([l_3, inputs[2]])

        # stage 4

        x = self.stage4_conv1_s(stage3_out)
        x = self.stage4_conv2_s(x)
        x = self.stage4_conv3_s(x)
        x = self.stage4_conv4_s(x)
        x = self.stage4_conv5_s(x)
        x = self.stage4_conv6_s(x)
        s_4 = self.stage4_conv7_s(x)
        x = self.stage4_conv1_l(stage3_out)
        x = self.stage4_conv2_l(x)
        x = self.stage4_conv3_l(x)
        x = self.stage4_conv4_l(x)
        x = self.stage4_conv5_l(x)
        x = self.stage4_conv6_l(x)
        l_4 = self.stage4_conv7_l(x)
        stage4_out = self.stage4_concat([s_4, l_4, vgg])

        if self.masked_outputs and inputs_with_masks:
            output_4_s = self.masked_4_s([s_4, inputs[1]])
            output_4_l = self.masked_4_l([l_4, inputs[2]])

        # stage 5

        x = self.stage5_conv1_s(stage4_out)
        x = self.stage5_conv2_s(x)
        x = self.stage5_conv3_s(x)
        x = self.stage5_conv4_s(x)
        x = self.stage5_conv5_s(x)
        x = self.stage5_conv6_s(x)
        s_5 = self.stage5_conv7_s(x)
        x = self.stage5_conv1_l(stage4_out)
        x = self.stage5_conv2_l(x)
        x = self.stage5_conv3_l(x)
        x = self.stage5_conv4_l(x)
        x = self.stage5_conv5_l(x)
        x = self.stage5_conv6_l(x)
        l_5 = self.stage5_conv7_l(x)
        stage5_out = self.stage5_concat([s_5, l_5, vgg])

        if self.masked_outputs and inputs_with_masks:
            output_5_s = self.masked_5_s([s_5, inputs[1]])
            output_5_l = self.masked_5_l([l_5, inputs[2]])

        # stage 6

        x = self.stage6_conv1_s(stage5_out)
        x = self.stage6_conv2_s(x)
        x = self.stage6_conv3_s(x)
        x = self.stage6_conv4_s(x)
        x = self.stage6_conv5_s(x)
        x = self.stage6_conv6_s(x)
        s_6 = self.stage6_conv7_s(x)
        x = self.stage6_conv1_l(stage5_out)
        x = self.stage6_conv2_l(x)
        x = self.stage6_conv3_l(x)
        x = self.stage6_conv4_l(x)
        x = self.stage6_conv5_l(x)
        x = self.stage6_conv6_l(x)
        l_6 = self.stage6_conv7_l(x)

        if self.masked_outputs and inputs_with_masks:
            output_6_s = self.masked_6_s([s_6, inputs[1]])
            output_6_l = self.masked_6_l([l_6, inputs[2]])

            return output_1_s, output_1_l, output_2_s, output_2_l, output_3_s, output_3_l, output_4_s, \
                   output_4_l, output_5_s, output_5_l, output_6_s, output_6_l
        else:
            return s_1, l_1, s_2, l_2, s_3, l_3, s_4, l_4, s_5, l_5, s_6, l_6
