#encoding:utf-8
### This script get the training labels and save them as TFRecords files
import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from utils.bbox.bbox import bbox_overlaps
from collections import Counter
sys.path.append(os.getcwd())
_stripe=16
OVERLAP_NEGATIVE_THR=0.5
OVERLAP_POSITIVE_THR=0.7
_heights=[11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
_widths=[16]

def gen_base_anchors():
    base_size=_stripe
    base_anchors=[]
    for h in _heights:
        for w in _widths:
            anchor=np.array([0,0,0,0],dtype=np.int)
            x_ctr=(0+base_size-1)/2
            y_ctr=(0+base_size-1)/2
            anchor[0]=x_ctr-w/2
            anchor[1]=y_ctr-h/2
            anchor[2]=x_ctr+w/2
            anchor[3]=y_ctr+h/2
            base_anchors.append(anchor)
    return np.stack(base_anchors)

def target_calc(anchors,gt_boxes,side_labels):
    # calculate the targets in vertical and target offset in herizotal for each anchor
    #inputs：anchors N*4, gt_boxes N*4
    #outputs：targets_y:N*2, targets_offset:N
    anchor_heigth=anchors[:,3]-anchors[:,1]+1
    anchor_y_ctr=(anchors[:,3]+anchors[:,1])/2

    gt_heigth=gt_boxes[:,3]-gt_boxes[:,1]+1
    gt_y_ctr=(gt_boxes[:,3]+gt_boxes[:,1])/2

    target_dy=(gt_y_ctr-anchor_y_ctr)/anchor_heigth
    target_dh=np.log(gt_heigth/anchor_heigth)
    target_y=np.stack([target_dy,target_dh]).transpose()

    target_offset=np.zeros(anchors.shape[0],dtype=np.float32)
    side_left_index=np.where(side_labels==1)[0]
    side_right_index = np.where(side_labels == -1)[0]
    target_offset[side_left_index]=(gt_boxes[side_left_index,0]-anchors[side_left_index,0])/_widths
    target_offset[side_right_index]=(gt_boxes[side_right_index,2]-anchors[side_right_index,2])/_widths

    return target_y,target_offset

def target_calc_inv(anchors,side_labels,targets_y,targets_offset):
    # calculate the proposal box from the targets_y and targets_offset
    # inputs：anchors N*4, 相对距离targets N*4
    # outputs：预测坐标 box N*4
    anchor_heigth = anchors[:, 3] - anchors[:, 1] + 1
    anchor_y_ctr = (anchors[:, 3] + anchors[:, 1]) / 2

    box_y_ctr=targets_y[:,0]*anchor_heigth+anchor_y_ctr
    box_height=np.exp(targets_y[:,1])*anchor_heigth
    box_x_left=anchors[:,0]+targets_offset*_widths
    box_x_right=anchors[:,2]+targets_offset*_widths

    box=np.zeros(anchors.shape,dtype=targets_y.dtype)
    box[:,1]=box_y_ctr-box_height/2
    box[:,3]=box_y_ctr+box_height/2
    box[:, 0] = anchors[:, 0]
    box[:, 2] = anchors[:, 2]
    side_left_index=np.where(side_labels==1)[0]
    box[side_left_index,0]=box_x_left[side_left_index] # left side offset refinement
    side_right_index=np.where(side_labels==-1)[0]
    box[side_right_index,2]=box_x_right[side_right_index] # right side offset refinement

    return box

def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[inds, :] = data
    return ret

def get_img_label(img_name,img_dir,gt_dir):
    # get the labels、targets for the input image
    name=img_name.split('.')[0]
    img_path=os.path.join(img_dir,img_name)
    gt_path=os.path.join(gt_dir,name+'.txt')
    img=cv2.imread(img_path)
    img_size=img.shape
    h_feat=img_size[0]//2//2//2//2  #feature map height
    w_feat=img_size[1]//2//2//2//2  #feature map width
    #get the image data in bytes
    with open(img_path,'rb') as ff:
        img_bytes=ff.read()
    ###read gt_box
    with open(gt_path,'r') as ff:
        lines=ff.readlines()
    box_list=[]
    side_flag_list=[]
    for line in lines:
        line=line.strip().split(',')
        side_flag_list.append(int(line[-1]))
        line = line[0:-1]
        x1,y1,x2,y2=map(int,line)
        box_list.append(np.array([x1,y1,x2,y2]))
    gt_boxes=np.stack(box_list)
    gt_side_flag=np.stack(side_flag_list)
    ###get all anchors
    base_anchors=gen_base_anchors()
    A=base_anchors.shape[0]
    K=h_feat*w_feat
    base_anchors=base_anchors.reshape(1,A,4)
    shift_x=np.arange(w_feat)*_stripe #对于feature map上每个点，x方向anchor偏移量
    shift_y=np.arange(h_feat)*_stripe #对于feature map上每个点，y方向anchor偏移量
    shift_x,shift_y=np.meshgrid(shift_x,shift_y) #生成二维点阵的x,y方向偏移量
    shift_x=shift_x.ravel() #二维变一维
    shift_y=shift_y.ravel()
    shift=np.stack([shift_x,shift_y,shift_x,shift_y]).transpose()
    shift=shift.reshape(K,1,4)
    all_anchors=base_anchors+shift
    all_anchors=all_anchors.reshape((K*A,4))
    total_anchors=K*A
    #remove the anchors that are out of the image
    index_inside=np.where(
        (all_anchors[:,0]>=0)&
        (all_anchors[:,1]>=0)&
        (all_anchors[:,2]<img_size[1])&
        (all_anchors[:,3]<img_size[0])
    )[0]
    anchors=all_anchors[index_inside,:]

    ### get labels, 1=positive, 0=negetive, -1=don't care
    labels=np.ones(anchors.shape[0],dtype=np.int)*-1
    labels2=np.ones(anchors.shape[0],dtype=np.int)*-1
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))
    max_ol_gt_index=overlaps.argmax(axis=1)
    max_ol_gt=overlaps[np.arange(anchors.shape[0]),max_ol_gt_index]
    max_ol_anchor_index=overlaps.argmax(axis=0)
    max_ol_anchor=overlaps[max_ol_anchor_index,np.arange(gt_boxes.shape[0])]

    labels[max_ol_gt<OVERLAP_NEGATIVE_THR]=0
    labels[max_ol_gt>=OVERLAP_POSITIVE_THR]=1
    labels2[max_ol_anchor_index]=1 #there is at least one positive anchor for each gt_box
    spectial_anchor_index=np.where((labels!=1)&(labels2==1))[0] #spectial anchor is the anchor that is the max overlap anchor of gt_box A,
                                                                # but this anchor's max overlaps with all the gt_box is gt_box B and <OVERLAP_NEGATIVE_THR
    spectial_anchor_gt_index=np.zeros(spectial_anchor_index.shape[0],dtype=np.int)
    for ii in range(spectial_anchor_gt_index.shape[0]):
        spectial_anchor_gt_index[ii]=np.where(max_ol_anchor_index==spectial_anchor_index[ii])[0][0]

    labels[spectial_anchor_index]=1

    ### side label:the anchor if is the left side or right side of the text area
    side_labels = np.zeros(anchors.shape[0], dtype=np.int)
    side_labels = gt_side_flag[max_ol_gt_index]
    side_labels[spectial_anchor_index]=gt_side_flag[spectial_anchor_gt_index]

    side_labels[np.where(labels!=1)[0]]=0

    ### get the targets
    gt_for_anchors=gt_boxes[max_ol_gt_index]
    gt_for_anchors[spectial_anchor_index]=gt_boxes[spectial_anchor_gt_index]
    box_targets_y,box_targets_offset=target_calc(anchors,gt_for_anchors,side_labels)
    box_weights_y=np.zeros((anchors.shape[0],2))
    box_weights_y[labels==1,:]=1
    box_targets_y=box_targets_y*box_weights_y

    ### get the anchor that are out of the image back
    labels=_unmap(labels,total_anchors,index_inside,fill=-1)
    side_labels=_unmap(side_labels,total_anchors,index_inside,fill=0)
    box_targets_y=_unmap(box_targets_y,total_anchors,index_inside,fill=0)
    box_targets_offset=_unmap(box_targets_offset,total_anchors,index_inside,fill=0)

    ### reshape
    labels=labels.reshape((1,h_feat,w_feat,A))
    side_labels=side_labels.reshape(1,h_feat,w_feat,A)
    box_targets_y=box_targets_y.reshape((1,h_feat,w_feat,A*2))
    box_targets_offset=box_targets_offset.reshape(1,h_feat,w_feat,A)

    return img_bytes,labels,side_labels,box_targets_y,box_targets_offset

def trans_labels(rpn_labels,rpn_side_labels,rpn_box_targets_y,rpn_box_target_offset):
    # recovery the image from labels,targets, to check if the labels and targets are correct
    h_feat, w_feat = rpn_labels.shape[1:3]
    base_anchors = gen_base_anchors()
    A = base_anchors.shape[0]
    K = h_feat * w_feat

    base_anchors = base_anchors.reshape(1, A, 4)
    shift_x = np.arange(w_feat) * _stripe  # 对于feature map上每个点，x方向anchor偏移量
    shift_y = np.arange(h_feat) * _stripe  # 对于feature map上每个点，y方向anchor偏移量
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)  # 生成二维点阵的x,y方向偏移量
    shift_x = shift_x.ravel()  # 二维变一维
    shift_y = shift_y.ravel()
    shift = np.stack([shift_x, shift_y, shift_x, shift_y]).transpose()
    shift = shift.reshape(K, 1, 4)
    all_anchors = base_anchors + shift
    all_anchors = all_anchors.reshape((K * A, 4))
    total_anchors = K * A
    side_labels=rpn_side_labels.reshape(total_anchors)
    targets_y=rpn_box_targets_y.reshape([-1,2])
    targets_offset=rpn_box_target_offset.reshape(total_anchors)
    boxes=target_calc_inv(all_anchors,side_labels,targets_y,targets_offset)
    labels=rpn_labels.reshape(total_anchors)
    boxes_fix=boxes[labels==1,:]
    return boxes_fix

def write_TFRecords(record_dir,img_dir,gt_dir):
    # save the image,labels,targets as TFRecords
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)
    img_name_list = os.listdir(img_dir)
    total_len =len(img_name_list)
    print('Total len of name list is:', total_len)
    next_index = 0  # the index of next name to be write
    while next_index<=total_len-1:
        batch_size=0
        name_list_left=img_name_list[next_index:]
        record_name='train'+str(next_index)+'.tfrecords'
        record_path=os.path.join(record_dir,record_name)
        with tf.io.TFRecordWriter(record_path) as writer:
            for name in name_list_left:
                img_bytes, rpn_labels,rpn_side_labels, rpn_targets_y,rpn_targets_offset = get_img_label(name,img_dir,gt_dir)
                label_shape = rpn_labels.shape
                ### check file size
                len_name = len(bytes(name, encoding='utf-8'))
                len_image = len(img_bytes)
                len_labels = rpn_labels.reshape(-1).shape[0]
                len_targets=len_labels*4
                len_label_shape = len(label_shape)
                len_sum = len_name + len_image + (len_labels+len_targets + len_label_shape) * 3  ## int and float calc 2.7bytes
                batch_size += len_sum
                if batch_size > 3e8:
                    next_index = img_name_list.index(name)
                    print('----------batch size is %d ------------' % batch_size)
                    break
                ### prepare to write, if batchsize<3e8
                features = {}
                features['image_name'] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[bytes(name, encoding='utf-8')]))
                features['image'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes]))
                features['labels'] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=rpn_labels.reshape(-1)))
                features['side_labels'] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=rpn_side_labels.reshape(-1)))
                features['targets_y'] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=rpn_targets_y.reshape(-1)))
                features['targets_offset'] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=rpn_targets_offset.reshape(-1)))
                features['label_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=label_shape))
                tf_features = tf.train.Features(feature=features)
                tf_example = tf.train.Example(features=tf_features)
                tf_serialized = tf_example.SerializeToString()
                writer.write(tf_serialized)
                next_index = img_name_list.index(name) + 1
                if next_index % 100 == 0:
                    print('Next index is %d' % next_index)
            print('batch_size=', batch_size)

def read_TFRecords(record_dir):
    # read from the TFRecords and recovery the image,to check if the TFRecords is correct
    file_list = os.listdir(record_dir)
    file_path_list = [os.path.join(record_dir, file_name) for file_name in file_list]
    raw_image_dataset = tf.data.TFRecordDataset(file_path_list)
    cnt_list=[]
    ee=0
    for raw_example in raw_image_dataset:
        if ee%100==0:
            print(ee)
        parsed=tf.train.Example.FromString(raw_example.numpy())
        image_name_feature=parsed.features.feature['image_name']
        image_feature=parsed.features.feature['image']
        labels_feature = parsed.features.feature['labels']
        side_labels_feature = parsed.features.feature['side_labels']
        targets_y_feature = parsed.features.feature['targets_y']
        targets_offset_feature = parsed.features.feature['targets_offset']
        label_shape_feature=parsed.features.feature['label_shape']
        image_name_bytes_recovery=image_name_feature.bytes_list.value[0]
        image_name_recovery=str(image_name_bytes_recovery,encoding='utf-8')
        image_bytes_recovery=image_feature.bytes_list.value[0]
        image_recovery = cv2.imdecode(np.frombuffer(image_bytes_recovery, np.uint8), cv2.IMREAD_COLOR)
        label_shape_recovery=label_shape_feature.int64_list.value
        labels_recovery = np.reshape(labels_feature.int64_list.value,label_shape_recovery)
        side_labels_recovery=np.reshape(side_labels_feature.int64_list.value,label_shape_recovery)
        target_y_shape=label_shape_recovery[0:3]+[label_shape_recovery[3]*2,]
        targets_y_recovery = np.reshape(targets_y_feature.float_list.value,target_y_shape)
        targets_offset_recovery=np.reshape(targets_offset_feature.float_list.value,label_shape_recovery)

        boxes = trans_labels(labels_recovery, side_labels_recovery,targets_y_recovery,targets_offset_recovery).astype(np.int)
        b_num = boxes.shape[0]
        for ii in range(b_num):
            cv2.rectangle(image_recovery, (boxes[ii, 0], boxes[ii, 1]), (boxes[ii, 2], boxes[ii, 3]), (0, 0, 255), 1)
        cv2.imshow(image_name_recovery, image_recovery)
        cv2.waitKey(0)

if __name__=='__main__':
    img_dir = 'training_data\\images'
    gt_dir = 'training_data\\gt_splited'
    record_dir='train_TFRecords'
    write_TFRecords(record_dir,img_dir,gt_dir)
    read_TFRecords(record_dir)