import argparse
import ctypes

import cv2
import numpy as np
import tensorrt as trt
import torch
import torch.nn as nn
from torchvision.transforms import Normalize


CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus",
               "train", "truck", "boat", "traffic_light", "fire_hydrant",
               "stop_sign", "parking_meter", "bench", "bird", "cat", "dog",
               "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
               "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
               "skis", "snowboard", "sports_ball", "kite", "baseball_bat",
               "baseball_glove", "skateboard", "surfboard", "tennis_racket",
               "bottle", "wine_glass", "cup", "fork", "knife", "spoon", "bowl",
               "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
               "hot_dog", "pizza", "donut", "cake", "chair", "couch",
               "potted_plant", "bed", "dining_table", "toilet", "tv", "laptop",
               "mouse", "remote", "keyboard", "cell_phone", "microwave",
               "oven", "toaster", "sink", "refrigerator", "book", "clock",
               "vase", "scissors", "teddy_bear", "hair_drier", "toothbrush"]

def args():

    args = argparse.ArgumentParser()
    args.add_argument('--plugin',type=str)
    args.add_argument('--engine',type=str)
    args.add_argument('--img',type=str)
    args.add_argument("--output",type=str)
    args.add_argument('--thresh',type=float,default=0.3)

    return args.parse_args()


def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device('cuda')
    elif device == trt.TensorLocation.HOST:
        return torch.device('cpu')
    else:
        return TypeError('%s is not supported by torch' % device)


def torch_dtype_from_trt(dtype):
    if dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int8:
        return torch.int8
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError('%s is not supported by torch' % dtype)


def cxcy2x1y1x2y2(bbox):

    x1y1 = bbox[...,:2] - bbox[...,2:] / 2
    x2y2 = bbox[...,:2] + bbox[...,2:] / 2

    return torch.cat([x1y1,x2y2],dim=-1)


class Preprocessimage(object):
    def __init__(self,h,w):
        self.inszie = (h,w)
        self.Normalize = Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225] ) 
		
    def __call__(self,image_path):
        raw_image = cv2.imread(image_path)
        image = cv2.cvtColor(raw_image,cv2.COLOR_BGR2RGB)
        H,W,_ = image.shape
        
        resize_facor = min(self.inszie[0]/H,self.inszie[1]/W)
        new_H,new_W = int(resize_facor * H) ,int(resize_facor * W)
        image = cv2.resize(image,(new_W,new_H),interpolation=cv2.INTER_LINEAR) 
        image = cv2.copyMakeBorder(image,0,self.inszie[0]-new_H,0,self.inszie[1]-new_W,cv2.BORDER_CONSTANT,value=(0,0,0))
        
        image = torch.from_numpy(image).float().cuda()
        image = image.permute(2,0,1) # chw
        image = self.Normalize(image/255.)
        image = image.unsqueeze(0)

        img_meta = dict()
        img_meta["image_path"] = image_path
        img_meta['scale'] = resize_facor
        img_meta["img_shape"] = (H,W)
        img_meta["input_shape"] = (self.inszie[0],self.inszie[1])

        return image,raw_image,img_meta


class TRTDeformDetr(nn.Module):

    def __init__(self,engine,max_per_img=100) -> None:
        super().__init__()
        self.max_per_img = max_per_img
        self._register_state_dict_hook(TRTDeformDetr._on_state_dict)
        self.TRT_logger = trt.Logger()
        trt.init_libnvinfer_plugins(self.TRT_logger,"")

        with open(engine,'rb') as f,\
            trt.Runtime(self.TRT_logger) as runtime:
            self.engine  = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
    
    def _on_state_dict(self,state_dict,prefix,local_metadata):
        state_dict[prefix+'engine'] = bytearray(self.engine.serialize())

    def forward(self,x:torch.Tensor,img_meta):
        num_img = x.shape[0]
        assert num_img==1, "now only support one img input"

        bindings = [None] * self.engine.num_bindings 
        bindings[0] = x.contiguous().data_ptr()

        outputs = [None] * 2

        output_index = {"bbox":1,"cls":0}
        for i in range(1,3):
            output_name = self.engine.get_binding_name(i)
            output_shape = tuple(self.context.get_binding_shape(i))
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(i))
            device = torch_device_from_trt(self.engine.get_location(i))

            output = torch.empty(size=output_shape,dtype=dtype,device=device)
            outputs[output_index[output_name]] = output
            bindings[i] = output.data_ptr()
        
        self.context.execute_async_v2(bindings,
                torch.cuda.current_stream().cuda_stream)

        cls_scores = outputs[0].sigmoid()
        num_classes = cls_scores.shape[-1]
        bbox_preds = outputs[1]

        cls_score = cls_scores[0]
        cls_score_per_img, top_indices = cls_score.flatten(0,1).topk(self.max_per_img,sorted=True)
        bbox_indices = top_indices // num_classes
        label_per_img = top_indices % num_classes

        bbox_pred_per_img = bbox_preds[0][bbox_indices]
        bbox_pred_per_img = cxcy2x1y1x2y2(bbox_pred_per_img)
        h,w = img_meta['input_shape']
        bbox_pred_per_img[...,0::2] *=w
        bbox_pred_per_img[...,1::2] *=h
        bbox_pred_per_img[...,0::2].clamp_(min=0,max=w)
        bbox_pred_per_img[...,1::2].clamp_(min=0,max=h)

        bbox_pred_per_img /= bbox_pred_per_img.new_tensor(img_meta['scale'])

        bboxes = np.zeros((self.max_per_img,5),dtype=np.float32)
        bboxes[...,-1] = cls_score_per_img.cpu().numpy()
        bboxes[...,0:4] = bbox_pred_per_img.cpu().numpy()

        bbox_results = [bboxes[label_per_img.cpu().numpy() ==i,:] for i in range(num_classes)]

        return bbox_results


if __name__=="__main__":

    opt = args()

    ctypes.CDLL(opt.plugin)

    color = [ [np.random.randint(0,255) for i in range(3)] for _ in range(len(CLASSES))]

    model = TRTDeformDetr(opt.engine,max_per_img=100)
    input_shape = model.context.get_binding_shape(0)
    perprocess = Preprocessimage(h=input_shape[2],w=input_shape[3])

    image,raw_image,img_meta = perprocess(opt.img)

    bbox_results = model(image,img_meta)
    bboxes = np.vstack(bbox_results)
    labels = [np.full(bbox.shape[0],i,dtype=np.int32) for i,bbox in enumerate(bbox_results)]

    labels = np.concatenate(labels)

    scores = bboxes[...,-1]

    inds = scores > opt.thresh

    labels = labels[inds]
    scores = scores[inds]
    bboxes = bboxes[...,0:4][inds,:]

    for i,(bbox,label,score) in enumerate(zip(bboxes,labels,scores)):
        bbox_int = bbox.astype(np.int32)
        c1,c2 = (int(bbox_int[0]),int(bbox_int[1])), (int(bbox_int[2]), int(bbox_int[3]))

        t1 = round(0.002 * (raw_image.shape[0] + raw_image.shape[1]) / 2) + 1

        cv2.rectangle(raw_image,c1,c2,color=color[label],thickness=t1,lineType=cv2.LINE_AA)
        cv2.putText(raw_image,CLASSES[label],c1,color=color[label],thickness=t1,fontFace=1,fontScale=2)

    cv2.imwrite(opt.output,raw_image)




        

