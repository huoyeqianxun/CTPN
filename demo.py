#encoding:utf-8
### This is a demo of the application of a trained CTPN model
import os
import proposal as pro
import proposal2poly as p2p
import tensorflow as tf
import numpy as np
import im2row
import cv2
import my_vgg16
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
WEIGHTS_PATH='vgg16\\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
class im2col_layer(tf.keras.layers.Layer):
    def __init__(self):
        super(im2col_layer, self).__init__()
    def call(self, conv_feat):
        # input: conv_feat from VGG conv5, shape is N*H*W*C
        # output: im2col result, block size is 3*3, so the shape after im2col is N*H*W*9C
        #         because N=1, reduce this dim, so the shape of out is H*W*9C
        out = im2row.im2row(conv_feat, (3, 3), padding='SAME')
        out = tf.squeeze(out, axis=0)
        return out

class CTPN_model2(tf.keras.Model):
    ###inputs:(1*img_h*img*w*3)
    ###outputs: x_cls_pre(h_feat,w_feat,10*2), x_box_pre(h_feat,w_feat,10*4)
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

if __name__=='__main__':
    data_dir='demo_data'
    res_dir='demo_result'
    file_list=os.listdir(data_dir)
    img_list=[file_name for file_name in file_list if file_name.split('.')[1]=='jpg' or file_name.split('.')[1]=='png']
    for img_name in img_list:
        img_path=os.path.join(data_dir,img_name)
        img=cv2.imread(img_path)
        img_size=img.shape[0:2]
        #resize
        ratio = 600 / min(img_size[0], img_size[1])
        if np.round(ratio * max(img_size[0], img_size[1])) > 1200:
            ratio = 1200 / max(img_size[0], img_size[1])
        H_new = int(img_size[0] * ratio)
        W_new = int(img_size[1] * ratio)
        H_new = H_new if H_new // 16 == 0 else (H_new // 16 + 1) * 16
        W_new = W_new if W_new // 16 == 0 else (W_new // 16 + 1) * 16  # otherwise ,the anchors cannot cover the whole img

        img_new = cv2.resize(img, (W_new, H_new))
        new_size = img_new.shape[0:2]

        img_tf = np.expand_dims(img_new, axis=0)
        img_tf = tf.cast(img_tf, tf.float32)
        img_tf = img_tf - tf.constant([117.75227555, 122.4191616, 127.7551814])
        my_model=CTPN_model2()
        my_model.load_weights('weights\\my_model_weights')
        cls_pre,box_pre_y,box_pre_offset=my_model(img_tf)

        scores,proposals,x_left_fixed,x_right_fixed=pro.proposal(cls_pre,box_pre_y,box_pre_offset,new_size)
        text_polys,score_polys=p2p.proposal2poly(proposals,scores,x_left_fixed,x_right_fixed,new_size,output_mode='poly')
        text_polys=text_polys.reshape(text_polys.shape[0],4,2)
        cv2.polylines(img_new, text_polys, 1, (0, 0, 255),1)
        img_res=cv2.resize(img_new,(img_size[1],img_size[0]))
        res_path=os.path.join(res_dir,img_name)
        cv2.imwrite(res_path,img_res)
        print('%s is done'%img_name)
    print('finished!')