import numpy as np
#计算混淆矩阵

class runningScore(object):
    '''
        n_classes: database的类别,包括背景
        ignore_index: 需要忽略的类别id,一般为未标注id, eg. CamVid.id_unlabel
    '''

    def __init__(self, n_classes, ignore_index=None):
        self.n_classes = n_classes
        #类别总数
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        # self.dice = 0
        # self.eps = 1e-5
        #存储混淆矩阵
        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")
        #要求忽视的坐标
        #np.bincount计算他的索引值在X中出现的次数
    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        #标签值应当大于等于0，小于种类数
        #设置bincount的最小长度为n为class的平方
        #将Bincount数组reshape成n_class*n_class
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
        #np.flatten将多维的数组降成一维
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)
        # for i in range(0, self.n_classes):
        #    intersection = np.sum((label_trues == i) * (label_preds == i))
        #    union = np.sum(label_trues == i) + np.sum(label_preds == i) + self.eps
        #    dice = 2.*intersection/union
        #    self.dice += dice


    def get_scores(self):
        """Returns accuracy score evaluation result.
            - pixel_acc:
            - class_acc: class mean acc
            - mIou :     mean intersection over union
            - fwIou:     frequency weighted intersection union
        """

        hist = self.confusion_matrix
        # total_dice = self.dice
        # n_classes = self.n_classes

        # ignore unlabel
        if self.ignore_index is not None:
            for index in self.ignore_index:
                hist = np.delete(hist, index, axis=0)
                hist = np.delete(hist, index, axis=1)
        #如果有需要忽视的种类，就在混淆矩阵中删除相应的行和列
        #np.diag以一维数组的方式返回方阵对角线元素
        #np.diag().sum()求对角线之和
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        #求出每个类别对应的像素准确率
        acc_cls = np.nanmean(acc_cls)
        #求出平均像素点准确率
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        #算出交并比,intersection over union
        mean_iou = np.nanmean(iu)
        #平均交并比
        freq = hist.sum(axis=1) / hist.sum()
        #求出每个类别出现的概率
        fw_iou = (freq[freq > 0] * iu[freq > 0]).sum()
        #求出加权交并比
        # dice =total_dice / n_classes

        # set unlabel as nan
        if self.ignore_index is not None:
            for index in self.ignore_index:
                iu = np.insert(iu, index, np.nan)
        #如果忽视的index非空，则需要在相应的位置插入nan
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "pixel_acc": acc,
                "class_acc": acc_cls,
                "mIou": mean_iou,
                "fwIou": fw_iou,
                # "dice": dice,
            },
            cls_iu,
        )
        #返回值分为两部分，第一部分为需要使用的
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count
