import matplotlib.pyplot as plt
import codecs
import math
import numpy as np


def open_tracked_oligo(filename_x, filename_y):
    with codecs.open(filename_x, encoding='utf-8-sig') as f1:
        tracked_oligo_x = np.loadtxt(f1)
    with codecs.open(filename_y, encoding='utf-8-sig') as f2:
        tracked_oligo_y = np.loadtxt(f2)
    
    return tracked_oligo_x, tracked_oligo_y


def find_correlation(period_1, period_2, curves_x_1, curves_y_1, curves_x_2, curves_y_2, curves_num_1, curves_num_2):
    correlation = np.zeros((curves_num_1, curves_num_2))
    distances = np.zeros(2000)
    pix_num = np.zeros(2000)
    mean_distances = []
    mean_dist_Rmax = 0
    for i in range(curves_num_1):
        for j in range(curves_num_2):
            Rmax = 0
            start_index_1 = i * period_1
            end_index_1 = np.max(np.nonzero(curves_x_1[start_index_1:start_index_1 + period_1])) + start_index_1
            start_index_2 = j * period_2
            end_index_2 = np.max(np.nonzero(curves_x_2[start_index_2:start_index_2 + period_2])) + start_index_2
            length_1 = end_index_1 - start_index_1 + 1
            length_2 = end_index_2 - start_index_2 + 1              
            if length_2 <= length_1:
                length = length_2
            else:
                length = length_1
            for k in range(abs(length_1 - length_2) + 1):
                distances = distances * 0
                pix_num = pix_num * 0
                for kk in range(length):
                    dx = curves_x_1[start_index_1 + kk + k] - curves_x_2[start_index_2 + kk]
                    dy = curves_y_1[start_index_1 + kk + k] - curves_y_2[start_index_2 + kk]
                    distances[kk] = math.sqrt(dx*dx + dy*dy)
                    pix_num[kk] = kk
                R = np.corrcoef(distances[0:kk + 1], pix_num[0:kk + 1])
                r = R[0, 1]
                mean_dist = distances[0:kk + 1].mean()
                if abs(r) >= Rmax:
                    Rmax = r
                    mean_dist_Rmax = mean_dist
                else:
                    pass

            if abs(Rmax) > 0.99999:
                mean_distances = np.append(mean_distances, [mean_dist_Rmax])
                correlation[i, j] = Rmax


    return correlation, mean_distances


if __name__ == "__main__":
    (curves_x_1, curves_y_1) = open_tracked_oligo('curve_x_1.txt', 'curve_y_1.txt')
    (curves_x_2, curves_y_2) = open_tracked_oligo('curve_x_2.txt', 'curve_y_2.txt')
    period_1 = curves_x_1.shape[1]
    period_2 = curves_x_2.shape[1]
    curves_num_1 = np.max(np.nonzero(curves_x_1[:, 0])) + 1
    curves_num_2 = np.max(np.nonzero(curves_x_2[:, 0])) + 1
    curves_x_1 = curves_x_1.ravel()
    curves_y_1 = curves_y_1.ravel()
    curves_x_2 = curves_x_2.ravel()
    curves_y_2 = curves_y_2.ravel()
    (correlation, mean_distances) = find_correlation(period_1, period_2, curves_x_1, curves_y_1, curves_x_2, curves_y_2, curves_num_1, curves_num_2)
    (mean_dist_hyst, mean_dist_bin) = np.histogram(mean_distances, bins=20, range = (0, 1000))
    correlation
