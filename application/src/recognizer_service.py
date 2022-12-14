import collections
import copy
import os
import string

import cv2
from glob import glob
import torch
from torch.nn import functional as F
import onnxruntime
import numpy as np
import time
import shortuuid
from scipy.special import softmax


class StrLabelConverter(object):

    def __init__(self, alphabet, ignore_case=True):

        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = '-' + alphabet

        self.alphabet_indicies = {char: i for i, char in enumerate(self.alphabet)}

    def decode(self, t, length, raw=False):

        if length.size == 1:
            length = length.item()
            if raw:
                return ''.join([self.alphabet[i] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i] == t[i - 1])):
                        index = t[i][0]
                        char_list.append(self.alphabet[index])
                return ''.join(char_list)
        else:
            # batch mode
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(self.decode(t[:, i], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


class LicensePlateRecognizer(object):
    def __init__(self, model_path):
        self.DEVICE_NAME = 'cpu'
        self.alphabet = string.digits + string.ascii_lowercase
        self.onnx_session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.converter = StrLabelConverter(self.alphabet)
        self.RECOGNIZER_IMG_CONFIGURATION = (0, 3, 1, 2)
        self.IMG_C = 3
        self.PIXEL_MAX_VALUE = 255
        self.recognizer_threshold = 0.85
        self.SEQUENCE_SIZE = 30
        self.ALPHABET = '-' + string.digits + string.ascii_lowercase
        self.BLANK_INDEX = 0
        print('Recognizer Model Onnx loaded')

    def run(self, image):
        img_reading_start_time = time.time()

        result = np.ascontiguousarray(
            np.stack([image]).astype(np.float32).transpose(self.RECOGNIZER_IMG_CONFIGURATION) / self.PIXEL_MAX_VALUE)
        ort_inputs = {self.onnx_session.get_inputs()[0].name: result}
        img_reading_end_time = time.time() - img_reading_start_time

        inference_running_start_time = time.time()
        test = self.onnx_session.run(None, ort_inputs)[0]
        output = softmax(test, axis=2)
        inference_running_end_time = time.time() - inference_running_start_time

        raw_labels = output.argmax(2)
        raw_probs = output.max(2)
        labels = []
        probs = []
        batch_size = 1

        for i in range(batch_size):
            current_prob = 1.0
            current_label = []
            for j in range(self.SEQUENCE_SIZE):

                if raw_labels[i][j] != self.BLANK_INDEX and not (j > 0 and raw_labels[i][j] == raw_labels[i][j - 1]):
                    current_label.append(self.ALPHABET[raw_labels[i][j]])
                    current_prob *= raw_probs[i][j]

            if not current_label:
                current_label.append(self.ALPHABET[self.BLANK_INDEX])
                current_prob = 0.0
            labels.append(''.join(current_label))
            probs.append(current_prob)
        print('\nrecognition time:')
        print(f'image reading time:{img_reading_end_time} seconds')
        print(f'inference time:{inference_running_end_time} seconds')
        return labels[0], probs[0]


if __name__ == '__main__':
    images = glob('./debug/*.jpeg') + glob('./debug/*.jpg')
    recognizer = LicensePlateRecognizer('./recognizer_base.onnx')
    for image_path in images:
        img_orig = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label, prob = recognizer.run(img_orig)
        print(image_path, label, prob)
