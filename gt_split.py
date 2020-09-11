#encoding:utf-8
### This script split the ground truth polys into boxes with width=16
import os
import sys
import cv2
import numpy as np
from shapely.geometry import Polygon

def pickTopLeft(poly):
    idx = np.argsort(poly[:, 0])
    if poly[idx[0], 1] < poly[idx[1], 1]:
        s = idx[0]
    else:
        s = idx[1]
    return poly[(s, (s + 1) % 4, (s + 2) % 4, (s + 3) % 4), :]

def orderConvex(p):
    ### rearange the order of the poly points, start from the left top, clockwise
    points = Polygon(p).convex_hull
    try:
        points = np.array(points.exterior.coords)[:4]
    except:
        points=np.array([0])
        return points
    points = points[::-1]
    points = pickTopLeft(points)
    points = np.array(points).reshape([4, 2])
    return points

def split_poly(poly,H, r=16):
    ### this function split the poly in gt_img_xxx.txt into polys(divided by anchor locations)
    #inputs: poly:4*2 np array; H,W:the image size; r: width of anchor
    #outputs:gt_boxes:(N*8)
    x_min = int(np.min(poly[:, 0]))
    x_max = int(np.max(poly[:, 0]))

    if (poly[1][0] - poly[0][0])==0 or (poly[2][0] - poly[3][0])==0:
        return np.array([])

    k1 = (poly[1][1] - poly[0][1]) / (poly[1][0] - poly[0][0])
    b1 = poly[0][1] - k1 * poly[0][0]

    k2 = (poly[2][1] - poly[3][1]) / (poly[2][0] - poly[3][0])
    b2 = poly[3][1] - k2 * poly[3][0]

    polys_splited = []

    start = int((x_min // 16 + 1) * 16)
    end = int((x_max // 16) * 16)

    if x_min < start - 1:  # filter the case x_min==start-1
        p = x_min
        polys_splited.append([p, max(int(k1 * p + b1),0),
                    start - 1, max(int(k1 * (p + 15) + b1),0),
                    start - 1, min(int(k2 * (p + 15) + b2),H),
                    p, min(int(k2 * p + b2),H)])

    for p in range(start, end, r):
        polys_splited.append([p, max(int(k1 * p + b1),0),
                    (p + 15), max(int(k1 * (p + 15) + b1),0),
                    (p + 15), min(int(k2 * (p + 15) + b2),H),
                    p, min(int(k2 * p + b2),H)])

    if x_max>end: # filter the case that x_max==end
        p=end
        polys_splited.append([p,max(int(k1 * p + b1),0),
                    x_max,max(int(k1 * x_max + b1),0),
                    x_max,min(int(k2 * x_max + b2),H),
                    p,min(int(k2 * p + b2),H)
        ])
    polys_splited=np.array(polys_splited, dtype=np.int).reshape([-1, 8])
    return polys_splited

def get_label_txt(file_path):
    #read the ground truth txt file,return the points list、language list、text list
    with open(file_path,'r',encoding='utf-8') as fp:
        lines=fp.readlines()
    coord_list=[]
    lan_list=[]
    text_list=[]
    for line in lines:
        line_temp=line.split(',')
        language = line_temp[8]
        text = line_temp[9]
        x1, y1, x2, y2, x3, y3, x4, y4 = map(float, line_temp[:8])
        coord = np.array([x1, y1, x2, y2, x3, y3, x4, y4]).reshape([4, 2])
        coord=orderConvex(coord)
        if len(coord.shape)==2:
            coord_list.append(coord)
            lan_list.append(language)
            text_list.append(text)
    return coord_list,lan_list,text_list

def get_training_splited_gt():
    ### This function read the training image and gt files, split, and save
    gt_dir = r'D:\MachineLearning\ICDAR data\ch8_training_localization_transcription_gt_v2' # replace your training gt files directory
    img_output_dir = 'training_data\\images'
    gt_output_dir = 'training_data\\gt_splited'
    if not os.path.exists(gt_output_dir):
        os.makedirs(gt_output_dir)
    if not os.path.exists(img_output_dir):
        os.makedirs(img_output_dir)

    cnt = 0
    for ii in range(1,9):
        img_dir=r'D:\MachineLearning\ICDAR data\ch8_training_images_'+str(ii) ## replace your training images directory
        img_name_list=os.listdir(img_dir)
        for img_name in img_name_list:
            if cnt % 100 == 0:
                print('cnt=', cnt, '---', img_name)
            cnt += 1
            img_type=img_name.split('.')[1]
            if img_type!='png' and img_type!='jpg': #cv2 can not read other type
                continue
            name = img_name.split('.')[0]
            gt_file_path = os.path.join(gt_dir, 'gt_' + name + '.txt')
            img_path = os.path.join(img_dir, img_name)

            ### img_resize, short side set to 600,and long side not longer than 1200
            img = cv2.imread(img_path)
            img_size = img.shape
            ratio=600/min(img_size[0],img_size[1])
            if np.round(ratio*max(img_size[0],img_size[1]))>1200:
                ratio=1200/max(img_size[0],img_size[1])
            H_new=int(img_size[0]*ratio)
            W_new=int(img_size[1]*ratio)
            H_new=H_new if H_new//16==0 else (H_new//16+1)*16
            W_new=W_new if W_new//16==0 else (W_new//16+1)*16 # otherwise ,the anchors cannot cover the whole img

            img_new = cv2.resize(img, (W_new, H_new))
            re_size=img_new.shape

            ###get the polys in gt_img_xxx.txt
            coord_list, lan_list, text_list = get_label_txt(gt_file_path)
            new_coord_list = []
            for coord in coord_list: #resize
                coord[:, 0] = coord[:, 0] * re_size[1] / img_size[1]
                coord[:, 1] = coord[:, 1] * re_size[0] / img_size[0]
                coord = coord.astype(np.int)
                new_coord_list.append(coord)

            box_list = []
            side_flag_list=[]#record if the box is the side box, 'L' means left side, 'R' means rigth side,'N' means not side
            for coord in new_coord_list:
                # drop the poly that size<10
                if np.linalg.norm(coord[0] - coord[1]) < 10 or np.linalg.norm(coord[3] - coord[0]) < 10:
                    continue
                polys = split_poly(coord, H_new,r=16)
                if polys.shape[0] == 0:
                    continue
                polys = polys.reshape([-1, 4, 2])
                for pp in range(polys.shape[0]):
                    x_min = np.min(polys[pp,:, 0])
                    y_min = np.min(polys[pp,:, 1])
                    x_max = np.max(polys[pp,:, 0])
                    y_max = np.max(polys[pp,:, 1])
                    box_list.append([x_min, y_min, x_max, y_max])
                    if pp == 0:
                        side_flag_list.append(1)  # side_flag: L=1, R=-1, not side=0
                    elif pp == polys.shape[0] - 1:
                        side_flag_list.append(-1)
                    else:
                        side_flag_list.append(0)
            if len(box_list) != 0:
                img_save_path = os.path.join(img_output_dir, img_name)
                cv2.imwrite(img_save_path, img_new)
                gt_output_path = os.path.join(gt_output_dir, name + '.txt')
                with open(gt_output_path, 'w') as ff:
                    for index,box in enumerate(box_list):
                        line = ",".join(str(box[i]) for i in range(4))
                        line=line+','+str(side_flag_list[index])
                        ff.writelines(line + '\n')

if __name__=="__main__":
    sys.path.append(os.getcwd())
    get_training_splited_gt()
    print('split finished!')

