import codecs
import numpy as np
import matplotlib.pyplot as plt


def open_merged_oligo(path_out_curve, num):
    filename_x = path_out_curve + "\\mt_curve_x_" + str(num) + ".txt"
    filename_y = path_out_curve + "\\mt_curve_y_" + str(num) + ".txt"
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


def curve_length(data, start_frame, end_frame):
    for j in range(start_frame, end_frame + 1):
        olig_x, olig_y = open_merged_oligo(path_out_curve, j)
        for i in range(olig_x.shape[0]):
            if not is_splitted(olig_x, olig_x[i, 0]):
                data[int(olig_x[i, 0]) - 1, j] = np.max(np.nonzero(olig_x[i, :]))
    return data


def is_splitted(olig_x, id):
    if len(list(np.where(olig_x[:, 0] == id)[0])) > 1:
        return True
    else:
        return False

def plot_curves(data, start_frame, end_frame):
    for i in range(data.shape[0]):
        if np.sum(data[i, 1:]) != 0:
            first_nonzero = np.min(np.nonzero(data[i, 1:]))
            last_nonzero = np.max(np.nonzero(data[i, 1:]))
            x = np.array(range(first_nonzero + 1, last_nonzero + 2))
            y = data[i, first_nonzero + 1 : last_nonzero + 2]
            plt.plot(x, y, color='k')
    plt.show()


path_out_curve = ".\\Data\\Output\\Merged_curves" 
data = np.zeros((200, 200))
data = np.hstack((np.array(range(1, data.shape[0] + 1)).reshape(data.shape[0], 1), data))
data = curve_length(data, start_frame = 6, end_frame = 13)
plot_curves(data, start_frame = 6, end_frame = 13)
