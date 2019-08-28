import numpy as np
import matplotlib.pyplot as plt

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy_ALLClass(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        return Acc

    def Pixel_Precision_ALLClass(self):
        Pr=np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=0)
        return Pr

    def F1_ALLClass(self):
        Acc=self.Pixel_Accuracy_ALLClass()
        Pr=self.Pixel_Precision_ALLClass()
        F1=(2*Acc*Pr)/(Acc+Pr)
        return F1

    def F1_MEANClass(self):
        Acc=self.Pixel_Accuracy_ALLClass()
        Pr=self.Pixel_Precision_ALLClass()
        F1=(2*Acc*Pr)/(Acc+Pr)
        F1_mean=np.nanmean(F1)
        return F1_mean

    def Class_Intersection_over_Union(self):
        IoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        return IoU

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def plot_confusion_matrix(self,epoch):        
        plt.imshow(self.confusion_matrix,interpolation='nearest',cmap=plt.cm.Paired)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks=np.arange(self.num_class)
        plt.xticks(tick_marks,tick_marks)
        plt.yticks(tick_marks,tick_marks)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('./%d.png'%epoch)






