# encoding:utf-8
### This script define the CTPN model and train it
import os
import sys
import gen_tfrecords
import tensorflow as tf
import numpy as np
import cv2
import time
import my_vgg16
import im2row
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#tf.config.run_functions_eagerly(True)
sys.path.append(os.getcwd())
WEIGHTS_PATH = r'vgg16\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
REGU_LOSS=tf.Variable(0,dtype=tf.float32)
LAMDA=2e-4 # weight decay

class im2col_layer(tf.keras.layers.Layer):
    def __init__(self):
        super(im2col_layer, self).__init__()
    def call(self, conv_feat):
        # input: conv_feat from VGG conv5, shape is N*H*W*C
        # output: im2col result, block size is 3*3, so the shape after im2col is N*H*W*9C
        #         because N=1, reduce this dim, so the shape of out is H*W*9C
        out = im2row.im2row(conv_feat, (3, 3), padding='SAME')                                                                        # script of tensorflow
        out = tf.squeeze(out, axis=0)
        return out

class CTPN_model2(tf.keras.Model):
    ###inputs(1*img_h*img*w*3)
    ###outputs: x_cls_pre(h_feat,w_feat,10*2), x_box_pre_y(h_feat,w_feat,10*2),x_box_pre_offset(h_feat,w_feat,10)
    def __init__(self):
        super(CTPN_model2, self).__init__(name='CTPN_model2')
        self.vgg16 = my_vgg16.VGG16(include_top=False, weights=WEIGHTS_PATH)
        self.im2col = im2col_layer()
        # #lstm layer
        initializer = tf.keras.initializers.VarianceScaling(scale=0.01, mode='fan_avg', distribution='truncated_normal')
        self.lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, input_dim=4608, kernel_initializer=initializer, return_sequences=True))
        # #FC layer
        self.FC = tf.keras.layers.Dense(512, activation='relu', kernel_initializer=initializer)
        # Text proposal
        self.cls_pre = tf.keras.layers.Dense(20, kernel_initializer=initializer)
        self.box_pre_y = tf.keras.layers.Dense(20, kernel_initializer=initializer)
        self.box_pre_offset=tf.keras.layers.Dense(10, kernel_initializer=initializer)

    def call(self, inputs):
        x = self.vgg16(inputs)
        x_im2col = self.im2col(x)
        x_lstm = self.lstm(x_im2col)
        x_FC = self.FC(x_lstm)
        x_cls_pre = self.cls_pre(x_FC)
        x_box_pre_y = self.box_pre_y(x_FC)
        x_box_pre_offset=self.box_pre_offset(x_FC)
        return x_cls_pre, x_box_pre_y,x_box_pre_offset

@tf.function(input_signature=(tf.TensorSpec(shape=([None, None, 20]), dtype=tf.float32),
                              tf.TensorSpec(shape=[None, None, 20], dtype=tf.float32),
                              tf.TensorSpec(shape=[None, None, 10], dtype=tf.float32),
                              tf.TensorSpec(shape=[1, None, None, 10], dtype=tf.int32),
                              tf.TensorSpec(shape=[1, None, None, 10], dtype=tf.int32),
                              tf.TensorSpec(shape=[1, None, None, 20], dtype=tf.float32),
                              tf.TensorSpec(shape=[1, None, None, 10], dtype=tf.float32),
                              ))
def loss_calc(cls_pre, box_pre_y, box_pre_offset,labels, side_labels, box_targets_y,box_targets_offset):
    print('in loss_calc')
    # calculate the loss

    ### cls loss
    cls_pre = tf.reshape(cls_pre, [-1, 2])  # reshape to HAW*2
    labels = tf.reshape(labels, [-1])
    side_labels=tf.reshape(side_labels,[-1])
    keep_index = tf.where(tf.not_equal(labels, -1))[:, 0]
    pos_index = tf.where(tf.equal(labels, 1))[:, 0]
    cls_pre_keep = tf.gather(cls_pre, keep_index)
    labels_keep = tf.gather(labels, keep_index)
    cls_loss_all = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_keep, logits=cls_pre_keep)
    cls_loss = tf.reduce_mean(cls_loss_all)

    ### box loss,only calc the positive anchors
    box_pre_y = tf.reshape(box_pre_y, [-1, 2])
    box_pre_offset=tf.reshape(box_pre_offset,[-1])
    box_targets_y = tf.reshape(box_targets_y, [-1, 2])
    box_targets_offset=tf.reshape(box_targets_offset,[-1])
    # delta of axis y, ie hight
    box_pre_keep_y = tf.gather(box_pre_y, pos_index)
    box_targets_keep_y = tf.gather(box_targets_y, pos_index)
    delta_y = box_pre_keep_y - box_targets_keep_y
    # smooth L1 loss of axis y
    smoothL1_sign_y = tf.cast(tf.less(tf.abs(delta_y), 1), tf.float32)
    delta_y_temp = 0.5 * tf.square(delta_y) * smoothL1_sign_y + (tf.abs(delta_y) - 0.5) * tf.abs(smoothL1_sign_y - 1)
    box_loss_y_temp = tf.reduce_mean(delta_y_temp, axis=0)

    ###loss of axis x, ie side offset
    #left side
    side_not_right_index=tf.where(tf.not_equal(side_labels, -1))[:, 0]
    num_left=tf.reduce_sum(tf.gather(side_labels,side_not_right_index))
    if num_left==0:# left side may be shut down by smampling
        box_loss_left_x_temp = tf.constant(0, tf.float32)
    else:
        side_left_index = tf.where(tf.equal(side_labels, 1))[:, 0]
        box_pre_left_keep = tf.gather(box_pre_offset, side_left_index)
        box_targets_left_keep = tf.gather(box_targets_offset, side_left_index)
        delta_left_x = box_pre_left_keep - box_targets_left_keep
        smoothL1_sign_left_x = tf.cast(tf.less(tf.abs(delta_left_x), 1), tf.float32)
        delta_left_x_temp = 0.5 * tf.square(delta_left_x) * smoothL1_sign_left_x + (tf.abs(delta_left_x) - 0.5) \
                            * tf.abs(smoothL1_sign_left_x - 1)
        box_loss_left_x_temp = tf.reduce_mean(delta_left_x_temp, axis=0)
    #right side
    side_not_left_index=tf.where(tf.not_equal(side_labels, 1))[:, 0]
    num_right = tf.reduce_sum(tf.gather(side_labels,side_not_left_index))
    if num_right==0:
        box_loss_right_x_temp = tf.constant(0, tf.float32)
    else:
        side_right_index = tf.where(tf.equal(side_labels, -1))[:, 0]
        box_pre_right_keep = tf.gather(box_pre_offset, side_right_index)
        box_targets_right_keep = tf.gather(box_targets_offset, side_right_index)
        delta_right_x=box_pre_right_keep-box_targets_right_keep
        smoothL1_sign_right_x = tf.cast(tf.less(tf.abs(delta_right_x), 1), tf.float32)
        delta_right_x_temp = 0.5 * tf.square(delta_right_x) * smoothL1_sign_right_x + (tf.abs(delta_right_x) - 0.5) \
                             * tf.abs(smoothL1_sign_right_x - 1)
        box_loss_right_x_temp = tf.reduce_mean(delta_right_x_temp, axis=0)
    box_loss = tf.reduce_sum(box_loss_y_temp)+2*box_loss_left_x_temp+2*box_loss_right_x_temp

    # regulation
    for var in my_model.trainable_variables:
        REGU_LOSS.assign_add(tf.nn.l2_loss(var))
    ###
    loss = cls_loss + box_loss+REGU_LOSS*LAMDA
    return loss, cls_loss, box_loss

@tf.function(input_signature=(tf.TensorSpec(shape=([None, None, 20]), dtype=tf.float32),
                              tf.TensorSpec(shape=[1, None, None, 10], dtype=tf.int32),
                              ))
def acc_num_calc(cls_pre, labels):
    ### calculate the statistics during training
    print('in acc_num_calc')
    cls_pre = tf.reshape(cls_pre, [-1, 2])  # reshape to HAW*2
    labels = tf.reshape(labels, [-1])
    pos_index = tf.where(tf.equal(labels, 1))[:, 0]
    neg_index = tf.where(tf.equal(labels, 0))[:, 0]

    num_pos_labels = tf.cast(tf.reduce_sum(tf.gather(labels, pos_index)), tf.float32)
    num_neg_labels = tf.cast(tf.reduce_sum(tf.gather(labels, neg_index) + 1), tf.float32)
    cls_softmax = tf.nn.softmax(cls_pre, axis=1)
    cls_pre_pos = tf.gather(cls_softmax, pos_index)[:, 1]
    num_pos_correct = tf.reduce_sum(tf.cast(tf.greater(cls_pre_pos, 0.7), tf.float32))  # label为1，且预测1的置信度
    cls_pre_neg = tf.gather(cls_softmax, neg_index)[:, 0]
    num_neg_correct = tf.reduce_sum(tf.cast(tf.greater(cls_pre_neg, 0.7), tf.float32))
    return num_pos_labels, num_pos_correct, num_neg_labels, num_neg_correct

@tf.function(input_signature=(tf.TensorSpec(shape=([1, None, None, 3]), dtype=tf.float32),
                              tf.TensorSpec(shape=([1, None, None, 10]), dtype=tf.int32),
                              tf.TensorSpec(shape=([1, None, None, 10]), dtype=tf.int32),
                              tf.TensorSpec(shape=([1, None, None, 20]), dtype=tf.float32),
                              tf.TensorSpec(shape=([1, None, None, 10]), dtype=tf.float32),
                              ))
def grad_calc(train_img, labels, side_labels, box_targets_y,box_targets_offset):
    print('in grad_calc')
    with tf.GradientTape() as tape:
        # print('forward calculating...')
        cls_pre, box_pre_y,box_pre_offset = my_model(train_img)
        # print('loss calculating...')
        loss, cls_loss, box_loss = loss_calc(cls_pre, box_pre_y, box_pre_offset,labels, side_labels, box_targets_y,box_targets_offset)
        # print('grad calculating...')
        grad = tape.gradient(loss, my_model.trainable_variables)
        num_pos_labels, num_pos_correct, num_neg_labels, num_neg_correct = acc_num_calc(cls_pre, labels)
        return loss, grad, cls_loss, box_loss, num_pos_labels, num_pos_correct, num_neg_labels, num_neg_correct

def measure_graph_size(f, *args):
    g = f.get_concrete_function(*args).graph
    print("{} contains {} nodes in its graph".format(
        f.__name__, len(g.as_graph_def().node)
    ))

def recovery_TFRecords(raw_example):
    ### read the TFRecords and recovery the image, labels and targets
    parsed = tf.train.Example.FromString(raw_example.numpy())
    image_name_feature = parsed.features.feature['image_name']
    image_feature = parsed.features.feature['image']
    labels_feature = parsed.features.feature['labels']
    side_labels_feature = parsed.features.feature['side_labels']
    targets_y_feature = parsed.features.feature['targets_y']
    targets_offset_feature = parsed.features.feature['targets_offset']
    label_shape_feature = parsed.features.feature['label_shape']
    image_name_bytes_recovery = image_name_feature.bytes_list.value[0]
    image_name_recovery = str(image_name_bytes_recovery, encoding='utf-8')
    image_bytes_recovery = image_feature.bytes_list.value[0]
    image_recovery = cv2.imdecode(np.frombuffer(image_bytes_recovery, np.uint8), cv2.IMREAD_COLOR)
    image_recovery = image_recovery.astype(np.float32) - np.array(
        [117.75227555, 122.4191616, 127.7551814])  # subtract the mean
    image_recovery = np.expand_dims(image_recovery, axis=0)
    label_shape_recovery = label_shape_feature.int64_list.value
    labels_recovery = np.reshape(labels_feature.int64_list.value, label_shape_recovery)
    side_labels_recovery = np.reshape(side_labels_feature.int64_list.value, label_shape_recovery)
    target_y_shape = label_shape_recovery[0:3] + [label_shape_recovery[3] * 2, ]
    targets_y_recovery = np.reshape(targets_y_feature.float_list.value, target_y_shape)
    targets_offset_recovery = np.reshape(targets_offset_feature.float_list.value, label_shape_recovery)
    return image_name_recovery, image_recovery, labels_recovery, side_labels_recovery,targets_y_recovery,targets_offset_recovery

def sampling(labels,side_labels):
    ### sample from the anchors
    num_limit = 1024
    num_pos_limit = 512
    labels_shape = labels.shape
    labels = labels.reshape(-1)
    side_labels = side_labels.reshape(-1)
    pos_index = np.where(labels == 1)[0]
    if len(pos_index) > num_pos_limit:
        disable_index = np.random.choice(pos_index, size=len(pos_index) - num_pos_limit, replace=False)
        labels[disable_index] = -1
    num_pos = len(np.where(labels == 1)[0])
    num_neg = num_limit - num_pos
    # num_neg=np.int(num_pos*1.2)
    neg_index = np.where(labels == 0)[0]
    if len(neg_index) > num_neg:
        disable_index = np.random.choice(neg_index, size=len(neg_index) - num_neg, replace=False)
        labels[disable_index] = -1
    num_neg = len(np.where(labels == 0)[0])
    side_labels[np.where(labels!=1)[0]]=0   #shut down the side_labels where labels!=1
    labels = labels.reshape(labels_shape)
    side_labels=side_labels.reshape(labels_shape)
    return labels,side_labels

def train():
    ### train the model
    record_dir = 'train_TFRecords'
    file_list = os.listdir(record_dir)
    file_path_list = [os.path.join(record_dir, file_name) for file_name in file_list]
    dataset = tf.data.TFRecordDataset(file_path_list)
    dataset = dataset.shuffle(buffer_size=7000)

    learning_rate = tf.Variable(1e-5, trainable=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    print(optimizer.get_config())
    ee = 0
    loss_list = []
    loss_cls_list = []
    loss_box_list = []
    num_pos_labels_list = []
    num_pos_correct_list = []
    num_neg_labels_list = []
    num_neg_correct_list = []

    for ii in range(8):
        for raw_example in dataset:
            #prepare the image and labels data
            starttime = time.time()
            img_name, img, labels_recovery, side_labels_recovery,targets_y,targets_offset = recovery_TFRecords(raw_example)
            labels,side_labels = sampling(labels_recovery,side_labels_recovery)
            labels = tf.cast(labels, tf.int32)
            side_labels=tf.cast(side_labels,tf.int32)
            targets_y = tf.cast(targets_y, tf.float32)
            targets_offset = tf.cast(targets_offset, tf.float32)
            img = tf.cast(img, tf.float32)
            REGU_LOSS.assign(0)
            # get the predication ,loss and grads
            loss, grads, cls_loss, box_loss, num_pos_labels, num_pos_correct, num_neg_labels, num_neg_correct \
                = grad_calc(img, labels, side_labels, targets_y,targets_offset)
            optimizer.apply_gradients(zip(grads, my_model.trainable_variables))
            # statistic
            loss_list.append(loss)
            loss_cls_list.append(cls_loss)
            loss_box_list.append(box_loss)
            num_pos_labels_list.append(num_pos_labels)
            num_pos_correct_list.append(num_pos_correct)
            num_neg_labels_list.append(num_neg_labels)
            num_neg_correct_list.append(num_neg_correct)
            if ee <= 100:
                loss_avg = sum(loss_list) / tf.convert_to_tensor(len(loss_list), tf.float32)
                loss_cls_avg = sum(loss_cls_list) / tf.convert_to_tensor(len(loss_cls_list), tf.float32)
                loss_box_avg = sum(loss_box_list) / tf.convert_to_tensor(len(loss_box_list), tf.float32)
                acc_pos = sum(num_pos_correct_list) / sum(num_pos_labels_list) * 100
                acc_neg = sum(num_neg_correct_list) / sum(num_neg_labels_list) * 100
                acc_all = (sum(num_pos_correct_list) + sum(num_neg_correct_list)) / (
                        sum(num_pos_labels_list) + sum(num_neg_labels_list)) * 100
            else:
                loss_avg = sum(loss_list[-100:]) / 100
                loss_cls_avg = sum(loss_cls_list[-100:]) / 100
                loss_box_avg = sum(loss_box_list[-100:]) / sum(loss_box_list[-100:])
                acc_pos = sum(num_pos_correct_list[-100:]) / sum(num_pos_labels_list[-100:]) * 100
                acc_neg = sum(num_neg_correct_list[-100:]) / sum(num_neg_labels_list[-100:]) * 100
                acc_all = (sum(num_pos_correct_list[-100:]) + sum(num_neg_correct_list[-100:])) / (
                        sum(num_pos_labels_list[-100:]) + sum(num_neg_labels_list[-100:])) * 100

            print('epoch:{}---loss:{:.3f}---loss_avg:{:.3f}---acc_pos:{:.2f}%---acc_neg:{:.2f}%---acc_all:{:.2f}%---t:{:.2f}s---name:{}'
                .format(ee, loss, loss_avg, acc_pos, acc_neg, acc_all, time.time() - starttime, img_name))

            if ee % 1000 == 0:# save the weights and statistic records
                loss_rec = [loss_list, loss_cls_list, loss_box_list]
                num_rec = [num_pos_correct_list, num_pos_labels_list, num_neg_correct_list, num_neg_labels_list]
                np.save('loss_rec.npy', np.stack(loss_rec))
                np.save('num_rec.npy', np.stack(num_rec))
                my_model.save_weights('my_model_weights')
            if ee % 3e4 == 0 and ee != 0: # change the learning rate from 30k epoches
                learning_rate.assign(1e-6)
                print(optimizer.get_config())
            ee += 1
        ii += 1
    loss_rec = [loss_list, loss_cls_list, loss_box_list]
    num_rec = [num_pos_correct_list, num_pos_labels_list, num_neg_correct_list, num_neg_labels_list]
    np.save('loss_rec.npy', np.stack(loss_rec))
    np.save('num_rec.npy', np.stack(num_rec))
    my_model.save_weights('my_model_weights')

if __name__ == '__main__':
    img_dir = 'training_data\\images'
    gt_dir = 'training_data\\gt_splited'
    img_bytes, labels, side_labels, box_targets_y,box_targets_offset = gen_tfrecords.get_img_label('img_1.png', img_dir, gt_dir)
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    img = np.expand_dims(img, axis=0)
    img = tf.cast(img, tf.float32)
    my_model = CTPN_model2()
    cls_pre, box_pre_y,box_pre_offset = my_model(img)
    train()



