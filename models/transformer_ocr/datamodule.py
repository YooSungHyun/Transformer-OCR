import os
import torch
import pytorch_lightning as pl
from argparse import Namespace
from utils.dataset_utils import get_dataset
from literal import Folder, DatasetColumns
from torch.nn.utils.rnn import pad_sequence
import cv2
import numpy as np
import numba as nb
from tokenizer import TransformerTokenizer


def remove_cursive_style(img):
    """Remove cursive writing style from image with deslanting algorithm"""

    def calc_y_alpha(vec):
        indices = np.where(vec > 0)[0]
        h_alpha = len(indices)

        if h_alpha > 0:
            delta_y_alpha = indices[h_alpha - 1] - indices[0] + 1

            if h_alpha == delta_y_alpha:
                return h_alpha * h_alpha
        return 0

    alpha_vals = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    rows, cols = img.shape
    results = []

    ret, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = otsu if ret < 127 else sauvola(img, (int(img.shape[0] / 2), int(img.shape[0] / 2)), 127, 1e-2)

    for alpha in alpha_vals:
        shift_x = max(-alpha * rows, 0.0)
        size = (cols + int(np.ceil(abs(alpha * rows))), rows)
        transform = np.asarray([[1, alpha, shift_x], [0, 1, 0]], dtype=np.float)

        shear_img = cv2.warpAffine(binary, transform, size, cv2.INTER_NEAREST)
        sum_alpha = 0
        sum_alpha += np.apply_along_axis(calc_y_alpha, 0, shear_img)
        results.append([np.sum(sum_alpha), size, transform])

    result = sorted(results, key=lambda x: x[0], reverse=True)[0]
    result = cv2.warpAffine(img, result[2], result[1], borderValue=255)
    result = cv2.resize(result, dsize=(cols, rows))

    return np.asarray(result, dtype=np.uint8)


def normalization(img):
    """Normalize list of image"""

    m, s = cv2.meanStdDev(img)
    img = img - m[0][0]
    img = img / s[0][0] if s[0][0] > 0 else img
    return img


def sauvola(img, window, thresh, k):
    """Sauvola binarization"""

    rows, cols = img.shape
    pad = int(np.floor(window[0] / 2))
    sum2, sqsum = cv2.integral2(cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT))

    isum = (
        sum2[window[0] : rows + window[0], window[1] : cols + window[1]]
        + sum2[0:rows, 0:cols]
        - sum2[window[0] : rows + window[0], 0:cols]
        - sum2[0:rows, window[1] : cols + window[1]]
    )

    isqsum = (
        sqsum[window[0] : rows + window[0], window[1] : cols + window[1]]
        + sqsum[0:rows, 0:cols]
        - sqsum[window[0] : rows + window[0], 0:cols]
        - sqsum[0:rows, window[1] : cols + window[1]]
    )

    ksize = window[0] * window[1]
    mean = isum / ksize
    std = (((isqsum / ksize) - (mean**2) / ksize) / ksize) ** 0.5
    threshold = (mean * (1 + k * (std / thresh - 1))) * (mean >= 100)

    return np.asarray(255 * (img >= threshold), "uint8")


@nb.jit(nopython=True)
def estimate_light_distribution(width, height, erosion, cei, int_img):
    """Light distribution performed by numba (thanks @Sundrops)"""

    for y in range(width):
        for x in range(height):
            if erosion[x][y] == 0:
                i = x

                while i < erosion.shape[0] and erosion[i][y] == 0:
                    i += 1

                end = i - 1
                n = end - x + 1

                if n <= 30:
                    h, e = [], []

                    for k in range(5):
                        if x - k >= 0:
                            h.append(cei[x - k][y])

                        if end + k < cei.shape[0]:
                            e.append(cei[end + k][y])

                    mpv_h, mpv_e = max(h), max(e)

                    for m in range(n):
                        int_img[x + m][y] = mpv_h + (m + 1) * ((mpv_e - mpv_h) / n)

                x = end
                break


def illumination_compensation(img, only_cei=False):
    """Illumination compensation technique for text image"""

    _, binary = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)

    if np.sum(binary) > np.sum(img) * 0.8:
        return np.asarray(img, dtype=np.uint8)

    def scale(img):
        s = np.max(img) - np.min(img)
        res = img / s
        res -= np.min(res)
        res *= 255
        return res

    img = img.astype(np.float32)
    height, width = img.shape
    sqrt_hw = np.sqrt(height * width)

    bins = np.arange(0, 300, 10)
    bins[26] = 255
    hp = np.histogram(img, bins)

    for i in range(len(hp[0])):
        if hp[0][i] > sqrt_hw:
            hr = i * 10
            break

    np.seterr(divide="ignore", invalid="ignore")
    cei = (img - (hr + 50 * 0.3)) * 2
    cei[cei > 255] = 255
    cei[cei < 0] = 0

    if only_cei:
        return np.asarray(cei, dtype=np.uint8)

    m1 = np.asarray([-1, 0, 1, -2, 0, 2, -1, 0, 1]).reshape((3, 3))
    m2 = np.asarray([-2, -1, 0, -1, 0, 1, 0, 1, 2]).reshape((3, 3))
    m3 = np.asarray([-1, -2, -1, 0, 0, 0, 1, 2, 1]).reshape((3, 3))
    m4 = np.asarray([0, 1, 2, -1, 0, 1, -2, -1, 0]).reshape((3, 3))

    eg1 = np.abs(cv2.filter2D(img, -1, m1))
    eg2 = np.abs(cv2.filter2D(img, -1, m2))
    eg3 = np.abs(cv2.filter2D(img, -1, m3))
    eg4 = np.abs(cv2.filter2D(img, -1, m4))

    eg_avg = scale((eg1 + eg2 + eg3 + eg4) / 4)

    h, w = eg_avg.shape
    eg_bin = np.zeros((h, w))
    eg_bin[eg_avg >= 30] = 255

    h, w = cei.shape
    cei_bin = np.zeros((h, w))
    cei_bin[cei >= 60] = 255

    h, w = eg_bin.shape
    tli = 255 * np.ones((h, w))
    tli[eg_bin == 255] = 0
    tli[cei_bin == 255] = 0

    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(tli, kernel, iterations=1)
    int_img = np.asarray(cei)

    estimate_light_distribution(width, height, erosion, cei, int_img)

    mean_filter = 1 / 121 * np.ones((11, 11), np.uint8)
    ldi = cv2.filter2D(scale(int_img), -1, mean_filter)

    result = np.divide(cei, ldi) * 260
    result[erosion != 0] *= 1.5
    result[result < 0] = 0
    result[result > 255] = 255

    return np.asarray(result, dtype=np.uint8)


def preprocess(img, input_size):
    """Make the process with the `input_size` to the scale resize"""
    img = np.array(img)

    if len(img.shape) == 3:
        if img.shape[2] == 4:
            trans_mask = img[:, :, 3] == 0
            img[trans_mask] = [255, 255, 255, 255]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    wt, ht, _ = input_size
    h, w = np.asarray(img).shape
    f = max((w / wt), (h / ht))
    new_size = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1))

    img = illumination_compensation(img)
    img = remove_cursive_style(img)
    img = cv2.resize(img, new_size)

    target = np.ones([ht, wt], dtype=np.uint8) * 255
    target[0 : new_size[1], 0 : new_size[0]] = img
    img = cv2.transpose(target)
    img = np.repeat(img[..., np.newaxis], 3, -1)
    img = normalization(img)  # w,h,c
    return img


class CustomDataLoader(torch.utils.data.DataLoader):
    def __init__(self, image_h, image_w, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(CustomDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.tokenizer = TransformerTokenizer()
        self.image_h = image_h
        self.image_w = image_w

    def _collate_fn(self, batch):
        images = np.array(
            [preprocess(feature[DatasetColumns.pixel_values], (self.image_w, self.image_h, 3)) for feature in batch]
        ).astype(np.float32)
        input_images = torch.from_numpy(images).permute(0, 3, 2, 1)  # B,C,H,W
        labels = None
        label_lengths = None
        if DatasetColumns.labels in batch[0]:
            texts = [torch.LongTensor(self.tokenizer.encode(feature[DatasetColumns.labels])) for feature in batch]
            labels = pad_sequence(texts, batch_first=True, padding_value=self.tokenizer.PAD)
            label_lengths = torch.IntTensor([len(text) for text in texts])
        return (input_images, labels, label_lengths)


class TransOCRDataModule(pl.LightningDataModule):
    def __init__(self, args: Namespace):
        super().__init__()
        self.train_data_path = args.train_data_path
        self.valid_data_path = args.valid_data_path
        self.test_data_path = args.test_data_path
        self.per_device_train_batch_size = args.per_device_train_batch_size
        self.per_device_eval_batch_size = args.per_device_eval_batch_size
        self.per_device_test_batch_size = args.per_device_test_batch_size
        self.num_workers = args.num_workers
        self.image_h = args.image_h
        self.image_w = args.image_w

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        if stage == "fit":
            self.train_datasets = get_dataset(os.path.join(Folder.data, self.train_data_path))
            self.valid_datasets = get_dataset(os.path.join(Folder.data, self.valid_data_path))
        if stage == "predict":
            self.test_datasets = get_dataset(os.path.join(Folder.data, self.test_data_path))

    def train_dataloader(self):
        return CustomDataLoader(
            dataset=self.train_datasets,
            batch_size=self.per_device_train_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            image_h=self.image_h,
            image_w=self.image_w,
        )

    def val_dataloader(self):
        return CustomDataLoader(
            dataset=self.valid_datasets,
            batch_size=self.per_device_train_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            image_h=self.image_h,
            image_w=self.image_w,
        )

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        return CustomDataLoader(
            dataset=self.test_datasets,
            batch_size=self.per_device_test_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            image_h=self.image_h,
            image_w=self.image_w,
            shuffle=False,
        )
