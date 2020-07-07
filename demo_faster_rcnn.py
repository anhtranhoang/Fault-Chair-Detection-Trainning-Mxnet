import cv2,os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'
os.environ['MXNET_GPU_MEM_POOL_ROUND_LINEAR_CUTOFF'] = '26'
os.environ['MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD'] = '999'
os.environ['MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD'] = '25'
os.environ['MXNET_GPU_COPY_NTHREADS'] = '8'
os.environ['MXNET_OPTIMIZER_AGGREGATION_SIZE'] = '54'
from gluoncv import model_zoo, data, utils
import matplotlib.pyplot as plt
import mxnet as mx
import glob,time
from random import shuffle
from predict_faster_rcnn import predict_product
from tqdm import tqdm


if __name__ == "__main__":
    ctx = mx.gpu(0)
    img = 'frame1'
    img = sorted(glob.glob(img + "/*"))
    path  = 'F:/HoangAnh/ScanComProject/Training/weightsfaster_rcnn_fpn_resnet50_v1b_coco_best.params'
    net = model_zoo.get_model('faster_rcnn_fpn_resnet50_v1b_coco', pretrained=False,ctx = [ctx])
    net.classes = ['_background_','back-left leg','back-right leg','front leg','box','hardware']
    net.load_parameters(path,allow_missing=True, ignore_extra=True,ctx = [ctx])
    # shuffle(img)
    
    i = 0    
    for im in tqdm(img):
        tic = time.time()
        ax,c = predict_product(net,im,ctx = ctx,mode = 1)
        # print("time process : ", time.time()-tic)
        cv2.imwrite("img/{s}.jpg".format(s = str(i).zfill(3)),ax)
        i += 1