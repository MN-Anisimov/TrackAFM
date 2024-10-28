import codecs
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def open_clicked_oligo(path_out_curve, num):
    filename_x = path_out_curve + "\\t_curve_x_" + str(num) + ".txt"
    filename_y = path_out_curve + "\\t_curve_y_" + str(num) + ".txt"
    with codecs.open(filename_x, encoding='utf-8-sig') as f:
        olig_x = np.loadtxt(f)
    with codecs.open(filename_y, encoding='utf-8-sig') as f:
        olig_y = np.loadtxt(f)
    if len(olig_x.shape) == 1:
        olig_x = olig_x.reshape(1, olig_x.shape[0])
        olig_y = olig_y.reshape(1, olig_y.shape[0])
    else:
        pass

    return olig_x, olig_y


def oligo_length(data, start_frame, end_frame):
    for j in range(start_frame, end_frame + 1):
        olig_x, olig_y = open_clicked_oligo(path_out_curve, j)
        for i in range(olig_x.shape[0]):
            data[int(olig_x[i, 0]) - 1, j] += np.max(np.nonzero(olig_x[i, :]))
    return data


def plot_curves(data, overlay):
    for i in range(data.shape[0]):
        if np.sum(data[i, 1:]) != 0:
            first_nonzero = np.min(np.nonzero(data[i, 1:]))
            last_nonzero = np.max(np.nonzero(data[i, 1:]))
            if last_nonzero - first_nonzero + 1 > 5:
                if overlay == False: x = np.array(range(first_nonzero + 1, last_nonzero + 2))
                else: x = np.array(range(1, last_nonzero - first_nonzero + 2))
                y = data[i, first_nonzero + 1 : last_nonzero + 2]            
                y = savgol_filter(y, 3, 1)
                plt.plot(x, y, color='k')
    plt.show()


path_out_curve = ".\\Data\\Output\\Curves" 
start_frame = 6
end_frame = 12
data = np.zeros((200, 200))
data = np.hstack((np.array(range(1, data.shape[0] + 1)).reshape(data.shape[0], 1), data))
data = oligo_length(data, start_frame, end_frame)
plot_curves(data, True)
