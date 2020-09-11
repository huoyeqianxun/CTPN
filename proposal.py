#encoding:utf-8
### This script calculate the proposals from the output of the CTPN model, and filter them with NMS
###
import numpy as np
import gen_tfrecords as gtf
import model_train as mtr
import tensorflow as tf
import cv2
from nms import nms
import os
import sys

def target_calc_inv_no_side_labels(anchors,targets_y,targets_offset):
    # get the proposal box from the output of the CTPN model
    # input：anchors N*4, targets_y N*4, targets_offset N*4
    # output：proposal boxes N*4
    anchor_heigth = anchors[:, 3] - anchors[:, 1] + 1
    anchor_y_ctr = (anchors[:, 3] + anchors[:, 1]) / 2

    box_y_ctr=targets_y[:,0]*anchor_heigth+anchor_y_ctr
    box_height=np.exp(targets_y[:,1])*anchor_heigth
    x_left_fixed=anchors[:,0]+targets_offset*gtf._widths
    x_right_fixed=anchors[:,2]+targets_offset*gtf._widths

    box=np.zeros(anchors.shape,dtype=targets_y.dtype)
    box[:,1]=box_y_ctr-box_height/2
    box[:,3]=box_y_ctr+box_height/2
    box[:, 0] = anchors[:, 0]
    box[:, 2] = anchors[:, 2]
    return box,x_left_fixed,x_right_fixed
def proposal(cls_pre,box_pre_y,box_pre_offset,img_size):
    ### calculate the proposals from the output of the CTPN model, and filter them with NMS
    ### input：cls_pre,box_pre_y,box_pre_offset：the output of the CTPN model；img_size:the original size of image
    ### output：proposals
    h_feat,w_feat=cls_pre.shape[0:2]
    K = h_feat * w_feat
    base_anchors = gtf.gen_base_anchors()
    A = base_anchors.shape[0]
    base_anchors = base_anchors.reshape(1, A, 4)
    shift_x = np.arange(w_feat) * gtf._stripe  # 对于feature map上每个点，x方向anchor偏移量
    shift_y = np.arange(h_feat) * gtf._stripe  # 对于feature map上每个点，y方向anchor偏移量
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)  # 生成二维点阵的x,y方向偏移量
    shift_x = shift_x.ravel()  # 二维变一维
    shift_y = shift_y.ravel()
    shift = np.stack([shift_x, shift_y, shift_x, shift_y]).transpose()
    shift = shift.reshape(K, 1, 4)
    all_anchors = base_anchors + shift
    all_anchors = all_anchors.reshape((K * A, 4))
    total_anchors = K * A

    box_pre_y=np.reshape(box_pre_y,[-1,2])
    box_pre_offset=np.reshape(box_pre_offset,-1)
    boxes,x_left_fixed,x_right_fixed=target_calc_inv_no_side_labels(all_anchors,box_pre_y,box_pre_offset)
    index_inside=np.where(
        (boxes[:,0]>=0) &
        (boxes[:,1]>=0) &
        (boxes[:,2]<img_size[1]) &
        (boxes[:,3]<img_size[0])
    )[0]
    proposals=boxes[index_inside,:]
    x_left_fixed=x_left_fixed[index_inside]
    x_right_fixed=x_right_fixed[index_inside]
    ###get the confidence scores for each anchor
    cls_pre = np.reshape(cls_pre, [-1, 2])
    cls_softmax = tf.nn.softmax(cls_pre, axis=1)
    scores=cls_softmax.numpy()[:,1]
    scores=scores[index_inside]
    ### keep the proposals with score>0.7
    index_keep=np.where(scores>0.7)[0]
    proposals=proposals[index_keep]
    x_left_fixed=x_left_fixed[index_keep]
    x_right_fixed=x_right_fixed[index_keep]
    scores=scores[index_keep]

    ###NMS filter
    order=scores.argsort()[::-1]
    proposals=proposals[order,:]
    scores=scores[order]
    x_left_fixed=x_left_fixed[order]
    x_right_fixed=x_right_fixed[order]
    scores=np.expand_dims(scores,axis=1)
    nms_input=np.hstack((proposals, scores)).astype(np.float32)
    nms_thresh=0.2
    keep = nms.nms(nms_input, nms_thresh)
    proposals=proposals[keep,:]
    x_left_fixed=x_left_fixed[keep]
    x_right_fixed=x_right_fixed[keep]
    scores=scores[keep]
    return scores,proposals,x_left_fixed,x_right_fixed

if __name__=='__main__':
    sys.path.append(os.getcwd())
    img_name='img_3.jpg'
    img_dir='validation_data\\images'
    gt_dir='validation_data\\gt_splited'
    img_path=os.path.join(img_dir,img_name)
    img=cv2.imread(img_path)
    img_size=img.shape[0:2]
    cls_pre=np.load('cls_pre.npy')
    box_pre_y=np.load('box_pre_y.npy')
    box_pre_offset=np.load('box_pre_offset.npy')

    _,labels,side_labels,box_targets_y,box_targets_offset=gtf.get_img_label(img_name,img_dir,gt_dir)
    num_pos_labels, num_pos_correct, num_neg_labels, num_neg_correct=mtr.acc_num_calc(cls_pre,labels)
    acc_pos=num_pos_correct/num_pos_labels*100
    acc_neg=num_neg_correct/num_neg_labels*100
    print("acc_pos:{:.2f}%---num_pos_correct:{}---num_pos_labels:{}".format(acc_pos,num_pos_correct,num_pos_labels))
    print("acc_neg:{:.2f}%---num_neg_correct:{}---num_neg_labels:{}".format(acc_neg,num_neg_correct,num_neg_labels))

    scores,proposals,x_left_fixed,x_right_fixed=proposal(cls_pre,box_pre_y,box_pre_offset,img_size)
    proposals=proposals.astype(np.int)
    p_num=proposals.shape[0]
    print('scores min:',scores.min())
    for ii in range(p_num):
        cv2.rectangle(img,(proposals[ii,0],proposals[ii,1]),(proposals[ii,2],proposals[ii,3]),(0,0,255),1)
    cv2.imshow('img_1',img)
    cv2.waitKey(0)
    print('finished!')
