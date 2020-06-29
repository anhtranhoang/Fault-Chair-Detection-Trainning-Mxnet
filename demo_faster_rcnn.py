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
ctx = mx.gpu(0)



path  = 'F:/HoangAnh/ScanComProject/Training/results/faster_rcnn_fpn_resnet50_v1b_coco_best.params'
net = model_zoo.get_model('faster_rcnn_fpn_resnet50_v1b_coco', pretrained=False,ctx = [ctx])
net.classes = ['_background_','back-left leg','back-right leg','front leg','top part','not top part']
net.load_parameters(path,allow_missing=True, ignore_extra=True,ctx = [ctx])
# img  = 'F:/HoangAnh/ScanComProject/Code/AppDemo/debug/1592187695.9447381_0.jpg'
img = 'debug'
img = glob.glob(img + "/*")
shuffle(img)
i = 0
from tqdm import tqdm
for im in tqdm(img):

    tic = time.time()
    x, orig_img = data.transforms.presets.rcnn.load_test(im,max_size= 1333,short=800)
    x = mx.nd.array(x,ctx = ctx)
    # orig_img = mx.nd.array(orig_img,ctx = ctx)
    # x, orig_img = data.transforms.presets.rcnn.transform_test()
    box_ids, scores, bboxes = net(x)
    ax,c = utils.viz.cv_plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], class_names=net.classes,thresh=0.4,mode = 1,debug = True)
    # print("process time : {}".format(time.time()-tic) )
    cv2.imwrite("img1/{}.jpg".format(i),ax)
    i += 1