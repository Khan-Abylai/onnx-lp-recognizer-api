import copy
import os
import cv2
from glob import glob
import torch
import onnxruntime
import numpy as np
import time
import shortuuid

from .config import cfg_mnet
from .utils import PriorBox, decode, decode_landm, py_cpu_nms


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class LicensePlateDetector(object):
    def __init__(self, model_path, image_width, image_height, det_threshold=0.7, nms_threshold=0.4, vis_threshold=0.7,
                 top_k=5000, keep_top_k=700):
        self.provider = ['CPUExecutionProvider']
        self.ort_session = onnxruntime.InferenceSession(model_path, providers=self.provider)
        self.img_w = image_width
        self.img_h = image_height
        self.mean = (104, 117, 123)
        self.transpose = (2, 0, 1)
        self.device = torch.device('cpu')
        self.det_threshold = det_threshold
        self.nms_threshold = nms_threshold
        self.vis_threshold = vis_threshold
        self.top_k = top_k
        self.keep_top_k = keep_top_k
        prior_box = PriorBox(cfg_mnet, image_size=(self.img_h, self.img_w))
        priors = prior_box.forward()
        priors = priors.to(torch.device('cpu'))
        self.prior_data = priors.data
        self.lp_ratio = 2.6
        self.PLATE_RECT = np.array([[0, 0], [0, 32], [127, 0], [127, 31]], dtype='float32')

        self.PLATE_SQUARE = np.array([[0, 0], [0, 63], [63, 0], [63, 63]], dtype='float32')

        self.RECOGNIZER_IMG_W = 128
        self.RECOGNIZER_IMG_H = 32
        self.RECOGNIZER_IMG_CONFIGURATION = (2, 0, 1)
        self.IMG_C = 3
        self.PIXEL_MAX_VALUE = 255

        print('License Plate Detector loaded into memory')

    def hconcat_resize_min(self, im_list, interpolation=cv2.INTER_CUBIC):
        h_min = min(im.shape[0] for im in im_list)
        im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                          for im in im_list]
        return cv2.hconcat(im_list_resize)

    def run(self, image):
        img_orig = copy.copy(image)
        main_start_time = time.time()
        start_time = time.time()
        img_raw = cv2.resize(image, (640, 480))
        img = np.float32(img_raw)
        im_height, im_width, _ = img_raw.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0)

        # compute ONNX Runtime output prediction
        ort_inputs = {self.ort_session.get_inputs()[0].name: to_numpy(img)}
        test = self.ort_session.run(None, ort_inputs)
        loc, conf, landms = test
        end_time = time.time() - start_time

        print(f'onnx exec time: {end_time}')

        start_time = time.time()
        device = torch.device("cpu")

        confidence_threshold = 0.7
        nms_threshold = 0.4
        vis_threshold = 0.7
        top_k = 10
        keep_top_k = 3

        loc = torch.Tensor(loc)
        loc = loc.to(device)
        conf2 = torch.Tensor(conf)
        conf2 = conf2.to(device)
        landms = torch.Tensor(landms)
        landms = landms.to(device)

        boxes = decode(loc.data.squeeze(0), self.prior_data, cfg_mnet['variance'])
        boxes = boxes * scale / 1
        boxes = boxes.cpu().numpy()

        scores = conf2.squeeze(0).data.cpu().numpy()[:, 1]
        landms2 = decode_landm(landms.data.squeeze(0), self.prior_data, cfg_mnet['variance'])
        scale1 = torch.Tensor(
            [img.shape[3], img.shape[2], img.shape[3], img.shape[2], img.shape[3], img.shape[2], img.shape[3],
             img.shape[2], img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms2 = landms2 * scale1 / 1
        landms2 = landms2.cpu().numpy()
        ## ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms2 = landms2[inds]
        scores = scores[inds]
        ##
        ## keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms2 = landms2[order]
        scores = scores[order]

        ## do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        end_time = time.time() - start_time
        print(f'preparing nms running:{end_time} seconds')
        start_time = time.time()
        keep = py_cpu_nms(dets, nms_threshold)

        ## keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms2 = landms2[keep]

        ## keep top-K faster NMS
        dets = dets[:keep_top_k, :]
        landms2 = landms2[:keep_top_k, :]

        dets = np.concatenate((dets, landms2), axis=1)

        end_time = time.time() - start_time
        print(f'nms running:{end_time} seconds')
        height, width, channels = image.shape
        sc_w = width / 640
        sc_h = height / 480

        main_end_time = time.time() - main_start_time
        print(f'exec main time:{main_end_time}')
        if len(dets) > 0:
            ind = np.argsort(dets[..., 4])
            key_points = dets[ind][0][5:].reshape(5, 2)

            key_points[:, ::2] *= sc_w
            key_points[:, 1::2] *= sc_h

            LT = key_points[0]
            RT = key_points[1]
            CP = key_points[2]
            LB = key_points[3]
            RB = key_points[4]

            plate_width = ((RT[0] - LT[0]) + (RB[0] - LB[0])) / 2
            plate_height = ((LB[1] - LT[1]) + (RB[1] - RT[1])) / 2

            ratio = plate_width / plate_height
            box_key_points = np.array([LT, LB, RT, RB])
            if ratio <= 2.6:
                plate_img = cv2.warpPerspective(img_orig,
                                                cv2.getPerspectiveTransform(box_key_points, self.PLATE_SQUARE),
                                                (int(self.RECOGNIZER_IMG_W // 2), self.RECOGNIZER_IMG_H * 2))
                top = plate_img[:32, :]
                bottom = plate_img[32:, :]
                plate_img = self.hconcat_resize_min([top, bottom])
            else:
                plate_img = cv2.warpPerspective(img_orig, cv2.getPerspectiveTransform(box_key_points, self.PLATE_RECT),
                                                (self.RECOGNIZER_IMG_W, self.RECOGNIZER_IMG_H))
            result = np.ascontiguousarray(
                plate_img.astype(np.float32).transpose(self.RECOGNIZER_IMG_CONFIGURATION) / self.PIXEL_MAX_VALUE)
            return result, plate_img, True
        else:
            return np.empty(shape=(3, 32, 128)), np.empty(shape=(32, 128, 3)), False


if __name__ == '__main__':
    detector = LicensePlateDetector('detector_mobilenet.onnx', 640, 480)

    images = glob('./images/*.jpeg') + glob('./images/*.jpg')

    for img_path in images:
        img_orig = cv2.imread(img_path, cv2.IMREAD_COLOR)
        inference_image, image, flag = detector.run(img_orig)
        stop = 1
        if flag:
            out_img_path = os.path.join('/home/kartykbayev/PycharmProjects/onnxRuntimeTest/debug',
                                        os.path.basename(img_path).replace('.jpeg', '-plate.jpeg').replace('.jpg',
                                                                                                           '-plate.jpg'))
            cv2.imwrite(out_img_path, image)
