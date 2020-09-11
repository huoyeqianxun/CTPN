#encoding:utf-8
### This script group the scripts and draw the poly or rectangle text area
import numpy as np

class GroupsGen():
    def __init__(self,proposals,scores,img_size):
        self.proposals = proposals
        self.scores = scores
        self.img_size = img_size
        self.heights = self.proposals[:, 3] - self.proposals[:, 1] + 1
    def meet_v_iou(self,index1,index2):
        h1=self.heights[index1]
        h2=self.heights[index2]
        y0=max(self.proposals[index1,1],self.proposals[index2,1])
        y1=min(self.proposals[index1,3],self.proposals[index2,3])
        over_height=max(0,y1-y0+1)
        overlaps=over_height/min(h1,h2)
        similarity=min(h1,h2)/max(h1,h2)
        if overlaps>=0.7 and similarity>=0.7:
            return True
        else:
            return False

    def get_right_adj(self,index):
        ###get the right adjacent proposal
        box=self.proposals[index]
        results=[]
        for pix in range(min(int(box[2])+1,self.img_size[1]-1),min(int(box[2])+1+32,self.img_size[1])):
            adj_box_indexes=self.boxes_table[pix]
            for adj_box_index in adj_box_indexes:
                if self.meet_v_iou(index,adj_box_index) and not(self.graph[:,adj_box_index].any()):#禁止被多个节点定义为右邻居
                    results.append(adj_box_index)
            if len(results)!=0:
                return results
        return results

    def get_left_adj(self,index):
        ###get the left adjacent proposal
        box=self.proposals[index]
        results=[]
        if int(box[0])==0:
            return results
        for pix in range(int(box[0])-1,max(int(box[0])-1-32,0)-1,-1):
            adj_box_indexes=self.boxes_table[pix]
            for adj_box_index in adj_box_indexes:
                if self.meet_v_iou(index,adj_box_index):
                    results.append(adj_box_index)
            if len(results)!=0:
                return results
        return results

    def is_conn_node(self,index,right_adj_index):
        ### for the proposal[index],if the left adjacent proposal of proposal[right_adj_index] is proposal[index] itself
        left_adjs=self.get_left_adj(right_adj_index)
        if self.scores[index]>=np.max(self.scores[left_adjs]):
            return True
        else:
            print("index:{}---right_adj_index:{}---find is_not_conn_node!!!".format(index,right_adj_index))
            return False

    def graph_build(self):
        ### build the neigbour graph
        boxes_table=[[] for _ in range(self.img_size[1])]
        for index,box in enumerate(self.proposals):
            boxes_table[int(box[0])].append(index)
        self.boxes_table=boxes_table

        self.graph=np.zeros((self.proposals.shape[0],self.proposals.shape[0]),np.bool)
        for index,box in enumerate(self.proposals):
            right_adjs=self.get_right_adj(index)
            if len(right_adjs)==0:
                continue
            right_adj_index=right_adjs[np.argmax(self.scores[right_adjs])]
            if self.is_conn_node(index,right_adj_index):
                self.graph[index,right_adj_index]=True

    def groups_gen(self):
        self.graph_build()
        ### group the proposals as the neighbour graph
        groups=[]
        for index in range(self.graph.shape[0]):
            if self.graph[index,:].any() and not(self.graph[:,index].any()):
            #若proposals[index]右侧有相邻proposal，而左侧没有，且找到一个group的起点
                node=index
                groups.append([node])
                while self.graph[node,:].any(): #如node右侧有相邻proposal
                    node=np.where(self.graph[node,:])[0][0]#node更新为右侧相邻proposal
                    groups[-1].append(node)
        return groups

def fit_y(X,Y,x1,x2):
    ### fit the line from points(X,Y), and return the (y1,y2) for (x1,x2)
    if np.sum(X==X[0])==len(X): #所有X坐标都一样
        return Y[0],Y[0]
    p=np.poly1d(np.polyfit(X,Y,1))
    return p(x1),p(x2)

def filter_poly(text_polys,score_polys):
    poly_num=text_polys.shape[0]
    heights=np.zeros((poly_num,1),np.float32)
    widths=np.zeros((poly_num,1),np.float32)
    for index,poly in enumerate(text_polys):
        heights[index]=(abs(poly[5]-poly[1])+abs(poly[7]-poly[3]))/2.0
        widths[index]=(abs(poly[2]-poly[0])+abs(poly[6]-poly[4]))/2.0
    ###filter:
    keep_index=np.where((widths/heights>=0.5)&(score_polys>0.9)&(widths>32))[0]
    return keep_index

def proposal2poly(proposals,scores,x_left_fixed,x_right_fixed,img_size,output_mode='rect'):
    #draw the text area from the grouped proposals
    # output_mode: 'rect'=draw rectangle；'poly'=draw poly
    groups_calc=GroupsGen(proposals,scores,img_size)
    groups=groups_calc.groups_gen()
    text_polys=np.zeros((len(groups),8),np.float32)
    score_polys=np.zeros((len(groups),1),np.float32)
    if output_mode=='rect':
        for index,group in enumerate(groups):
            boxes=proposals[group]
            x0= x_left_fixed[group][0] #每组x方向的左、右边界，采用经offset修正过的坐标
            x1 = x_right_fixed[group][-1]
            lt_y, rt_y = fit_y(boxes[:, 0], boxes[:, 1], x0, x1)  # 计算左上、右上y值
            lb_y, rb_y = fit_y(boxes[:, 2], boxes[:, 3], x0, x1)  # 计算左下、右下y值
            score_polys[index]=scores[group].sum()/float(len(group)) #计算本组平均分
            text_polys[index,0]=x0
            text_polys[index,1]=min(lt_y,rt_y)  #这里实际是矩形，目的是尽量少的切割掉文字，而不是与label poly的吻合最佳
            text_polys[index,2]=x1
            text_polys[index,3]=min(lt_y,rt_y)
            text_polys[index,4]=x1
            text_polys[index,5]=max(rb_y,lb_y)
            text_polys[index,6]=x0
            text_polys[index,7]=max(rb_y,lb_y)
    else:
        for index,group in enumerate(groups):
            boxes=proposals[group]
            x0= x_left_fixed[group][0] #每组x方向的左、右边界，采用经offset修正过的坐标
            x1 = x_right_fixed[group][-1]
            lt_y, rt_y = fit_y(boxes[:, 0], boxes[:, 1], x0, x1)  # 计算左上、右上y值
            lb_y, rb_y = fit_y(boxes[:, 2], boxes[:, 3], x0, x1)  # 计算左下、右下y值
            score_polys[index]=scores[group].sum()/float(len(group)) #计算本组平均分
            text_polys[index,0]=x0
            text_polys[index,1]=lt_y
            text_polys[index,2]=x1
            text_polys[index,3]=rt_y
            text_polys[index,4]=x1
            text_polys[index,5]=rb_y
            text_polys[index,6]=x0
            text_polys[index,7]=lb_y
    keep_index=filter_poly(text_polys,score_polys)
    text_polys=text_polys[keep_index]
    score_polys=score_polys[keep_index]
    text_polys=text_polys.astype(np.int)
    return text_polys,score_polys