import os

import cv2
import mxnet as mx
from gluoncv import data, model_zoo, utils


def predict_product(net,img,mode,ctx = None):
    """
    Predict product on given image and return results in unique mode.

        Parameters
        ----------
        net : Network with loading parameters.
        img : Given image
            numpy.ndarray or mxnet.nd.NDArray
            Image with shape `H, W, 3` or name of image (path).
        ctx : context.
        mode : image compatible with camera
            For example: 
                - If camera in a side view, mode = 'side' else mode = 'top'.
        Return 
        ----------
        ax : processed image with rectangle for each object and its accuracy.
        c : class of object which's in image.

    """
    if ctx == None :
        ctx = mx.cpu()
    else :
        ctx = ctx
    if isinstance(img,str):
        x, orig_img = data.transforms.presets.rcnn.load_test(img,max_size= 1333,short=800)
    else :
        x, orig_img = data.transforms.presets.rcnn.transform_test(img,max_size= 1333,short=800)

    x = mx.nd.array(x,ctx = ctx)
    box_ids, scores, bboxes = net(x)
    ax,c= utils.viz.cv_plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], 
                                    class_names=net.classes,thresh=0.75,
                                        mode = mode)

    return ax,c