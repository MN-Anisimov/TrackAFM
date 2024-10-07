import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import codecs
from matplotlib.backend_bases import MouseButton
from numpy.ma import masked_array
from pathlib import Path
import copy


def open_curve_from_file(path1, path2, name_x, name_y, num, mode):
    global new_id
    file = Path(path1 + "t_" + name_x + str(num)  + ".txt")
    if file.is_file():
        with codecs.open(path1 + "t_" + name_x + str(num)  + ".txt", encoding='utf-8-sig') as f:
            curve_x = np.loadtxt(f)
        with codecs.open(path1 + "t_" + name_y + str(num)  + ".txt", encoding='utf-8-sig') as f:
            curve_y = np.loadtxt(f)
        if len(curve_x.shape) == 1:
            tot_olig_num = 1
            curve_x = curve_x.reshape(1, curve_x.shape[0])
            curve_y = curve_y.reshape(1, curve_y.shape[0])
        else:
            tot_olig_num = curve_x.shape[0]
        new_id = np.max(curve_x[:, 0])
    else:
        with codecs.open(path2 + name_x + str(num)  + ".txt", encoding='utf-8-sig') as f:
            curve_x = np.loadtxt(f)
        with codecs.open(path2 + name_y + str(num)  + ".txt", encoding='utf-8-sig') as f:
            curve_y = np.loadtxt(f)

        curve_x, curve_y, tot_olig_num = numerate_curves(curve_x, curve_y, mode)

    return curve_x, curve_y, tot_olig_num


def open_im(path, name, num):
    with codecs.open(path + name + str(num)  + ".txt", encoding='utf-8-sig') as f:
        im = np.loadtxt(f)
    
    return im


def write_curve_to_file(path, name_x, name_y, num, curve_x, curve_y):
    file = Path(path + name_x + str(num)  + ".txt")
    if not file.is_file():
        curve_x, curve_y = curve_x[~np.all(curve_x == 0, axis=1)], curve_y[~np.all(curve_y == 0, axis=1)]
        np.savetxt(path + name_x + str(num) + ".txt", curve_x, '%d', encoding = 'utf-8-sig')
        np.savetxt(path + name_y + str(num) + ".txt", curve_y, '%d', encoding = 'utf-8-sig')
    else:
        pass


def numerate_curves(curve_x, curve_y, mode):
    if len(curve_x.shape) == 1:
        tot_olig_num = 1
        curve_x = curve_x.reshape(1, curve_x.shape[0])
        curve_y = curve_y.reshape(1, curve_y.shape[0])
    else:
        tot_olig_num = curve_x.shape[0]

    if mode == 1:
        curve_x = np.hstack((np.array(range(1, tot_olig_num + 1)).reshape(tot_olig_num, 1), curve_x))
        curve_y = np.hstack((np.array(range(1, tot_olig_num + 1)).reshape(tot_olig_num, 1), curve_y))
    else:
        curve_x = np.hstack((np.zeros((1, tot_olig_num)).reshape(tot_olig_num, 1), curve_x))
        curve_y = np.hstack((np.zeros((1, tot_olig_num)).reshape(tot_olig_num, 1), curve_y))

    return curve_x, curve_y, tot_olig_num


def find_olig(x0, y0, w, what_fig):
    if x0 < w:
        x1 = 0
    else:
        x1 = x0 - w

    if x_max - x0 < w:
        x2 = x_max - 1
    else:
        x2 = x0 + w

    if y0 < w:
        y1 = 0
    else:
        y1 = y0 - w

    if y_max - y0 < w:
        y2 = y_max - 1
    else:
        y2 = y0 + w
    
    list_x = [_ + y0 * x_max + x1 for _ in range(x2 - x1 + 1)]
    list_y = [y1 * x_max + x0 + x_max * _ for _ in range(y2 - y1 + 1)]

    if what_fig == 1:
        if np.any(data_field_1[list_x]) or np.any(data_field_1[list_y]):
            if np.any(data_field_1[list_x]):
                index = list_x[int(np.min(np.nonzero(data_field_1[list_x])))]
            else:
                index = list_y[int(np.min(np.nonzero(data_field_1[list_y])))]
            return index % x_max, index // x_max
        else:
            return None, None
    else:
        if np.any(data_field_2[list_x]) or np.any(data_field_2[list_y]):
            if np.any(data_field_2[list_x]):
                index = list_x[int(np.min(np.nonzero(data_field_2[list_x])))]
            else:
                index = list_y[int(np.min(np.nonzero(data_field_2[list_y])))]
            return index % x_max, index // x_max
        else:
            return None, None


def next_image():
    global im_1
    global im_2
    global im_num
    global curve_x_1
    global curve_y_1
    global curve_x_2
    global curve_y_2
    global tot_olig_num_1
    global tot_olig_num_2
    global data_field_1
    global data_field_2

    data_field_1 = data_field_1 * 0
    data_field_2 = data_field_2 * 0
    curve_x_1 = curve_x_2
    curve_y_1 = curve_y_2
    tot_olig_num_1 = tot_olig_num_2
    im_1 = copy.deepcopy(im_2)

    curve_x_2, curve_y_2, tot_olig_num_2 = open_curve_from_file(path_inp_curve_1, path_inp_curve_2, "curve_x_", "curve_y_", im_num + 1, 2)
    im_2 = open_im(path_inp_im, "image_", im_num + 1)
    
    im_num += 1


def plot_im(im):
     plt.matshow(masked_array(im, im > 0.5), cmap=matplotlib.cm.Greys_r, alpha = 0.5, fignum = 0)
     plt.matshow(masked_array(im, im != 1), cmap=ListedColormap(["blue"]), fignum = 0)
     plt.matshow(masked_array(im, im != 2), cmap=ListedColormap(["green"]), fignum = 0)
     plt.matshow(masked_array(im, im != 3), cmap=ListedColormap(["red"]), fignum = 0)


def add_all_olig():
    global data_field_1
    global data_field_2
    global im_1
    global im_2
    global im_1_to_plot
    global im_2_to_plot
    global im_2_to_plot_2

    im_1_to_plot = copy.deepcopy(im_1)
    im_2_to_plot = copy.deepcopy(im_2)
    for i in range(tot_olig_num_1):
        if np.any(curve_x_1[i, 1:] != 0):
            indexes = curve_y_1[i, np.nonzero(curve_y_1[i, 1:])[0] + 1] * x_max + curve_x_1[i, np.nonzero(curve_x_1[i, 1:])[0] + 1]
            np.put(data_field_1, list(indexes), 1) # data_field must be 1D array
            im_1_to_plot = im_1_to_plot + data_field_1.reshape((x_max, y_max))
            im_1_to_plot[im_1_to_plot > 0.5] = 1
        else:
            pass

    for i in range(tot_olig_num_2):
        indexes = curve_y_2[i, np.nonzero(curve_y_2[i, 1:])[0] + 1] * x_max + curve_x_2[i, np.nonzero(curve_x_2[i, 1:])[0] + 1]
        np.put(data_field_2, list(indexes), 1) # data_field must be 1D array
        im_2_to_plot = im_2_to_plot + data_field_2.reshape((x_max, y_max))
        im_2_to_plot[im_2_to_plot > 0.5] = 1


def fig2_add_olig():
    global olig_num_2
    global data_field_2
    global im_2_to_plot_2
    global im_2_to_plot
    global copy_data_field_2
    global sum_data_field_2

    indexes = curve_y_2[olig_num_2, np.nonzero(curve_y_2[olig_num_2, 1:])[0] + 1] * x_max + curve_x_2[olig_num_2, np.nonzero(curve_x_2[olig_num_2, 1:])[0] + 1]
    copy_data_field_2 = copy_data_field_2 * 0
    np.put(copy_data_field_2, list(indexes), 1)
    np.put(sum_data_field_2, list(indexes), 1)
    im_2_to_plot_2 = im_2_to_plot + copy_data_field_2.reshape((x_max, y_max))
    im_2_to_plot_2[im_2_to_plot_2 > 1.5] = 2
    im_2_to_plot_2 = im_2_to_plot_2 + sum_data_field_2.reshape((x_max, y_max))
    im_2_to_plot_2

def compare_curves(curve_x_get, curve_y_get, curve_x_set, curve_y_set, x, y, mode):
    global olig_num_2
    global new_id
    curve_eq_x = (curve_x_set[:, 1:3] == curve_x_set[olig_num_2, 1:3]) 
    curve_eq_y = (curve_y_set[:, 1:3] == curve_y_set[olig_num_2, 1:3])
    rows_get = np.where((curve_x_get[:, 1:] == x) & (curve_y_get[:, 1:] == y))[0]
    rows_set = np.where(curve_eq_x[:, 0] * curve_eq_y[:, 1])[0]
    if rows_get.size == 0: return None, None
    elif rows_get.size > rows_set.size: return None, None
    elif mode == 3:
        curve_x_set[olig_num_2, :] = 0
        curve_y_set[olig_num_2, :] = 0
        olig_num_2 += 1
        return 0, olig_num_2 - 1
    elif rows_get.size == rows_set.size == 1:
        row_get = rows_get[0]
        same_get_set_x = curve_x_get[row_get, 1:3] == curve_x_set[olig_num_2, 1:3]
        same_get_set_y = curve_y_get[row_get, 1:3] == curve_y_set[olig_num_2, 1:3]
        the_same = np.sum(same_get_set_x * same_get_set_y)
        if the_same and (mode == 2):
            if new_id == 0: new_id = tot_olig_num_1 + 1
            else: new_id += 1                    
            curve_x_set[olig_num_2, 0] = new_id
            curve_y_set[olig_num_2, 0] = new_id
            olig_num_2 += 1
            return 0, olig_num_2 - 1
        else:
            if curve_x_get[row_get, 0] == 0: return None, None
            else:
                curve_x_set[olig_num_2, 0] = curve_x_get[row_get, 0]
                curve_y_set[olig_num_2, 0] = curve_y_get[row_get, 0]
                olig_num_2 += 1
                return 0, olig_num_2 - 1                
    else:
        if rows_get.size == 2:
            row_set1 = rows_set[0]
            row_set2 = rows_set[1]
            row_get1 = rows_get[0]
            row_get2 = rows_get[1]
        else:
            row_set1 = rows_set[0]
            row_set2 = rows_set[1]
            row_get1 = rows_get[0]
            row_get2 = rows_get[0]
        same_get_set_x = curve_x_get[row_get1, 1:3] == curve_x_set[olig_num_2, 1:3]
        same_get_set_y = curve_y_get[row_get1, 1:3] == curve_y_set[olig_num_2, 1:3]
        the_same = np.sum(same_get_set_x * same_get_set_y)
        if the_same and (mode == 2):
            if new_id == 0: new_id = tot_olig_num_1 + 1
            else: new_id += 1                    
            curve_x_set[olig_num_2, 0] = new_id
            curve_y_set[olig_num_2, 0] = new_id
            olig_num_2 += 1
            return 0, olig_num_2 - 1
        else:
            if curve_x_get[row_get1, 0] == curve_x_get[row_get2, 0] == 0: return None, None
            elif curve_x_set[row_set1, 0] == curve_x_set[row_set2, 0] == 0:
                    curve_x_set[olig_num_2, 0] = curve_x_get[row_get1, 0]
                    curve_y_set[olig_num_2, 0] = curve_y_get[row_get1, 0]
                    olig_num_2 += 1
                    return 0, olig_num_2 - 1
            elif (curve_x_set[row_set1, 0] == 0) and (curve_x_set[row_set2, 0] != 0): #make shorter
                if curve_x_set[row_set2, 0] != curve_x_get[row_get1, 0]:
                    curve_x_set[olig_num_2, 0] = curve_x_get[row_get1, 0]
                    curve_y_set[olig_num_2, 0] = curve_y_get[row_get1, 0]
                    olig_num_2 += 1
                    return 0, olig_num_2 - 1
                elif rows_get.size == 2: 
                    curve_x_set[olig_num_2, 0] = curve_x_get[row_get2, 0]
                    curve_y_set[olig_num_2, 0] = curve_y_get[row_get2, 0]
                    olig_num_2 += 1
                    return 0, olig_num_2 - 1
                else: return None, None
            elif (curve_x_set[row_set1, 0] != 0) and (curve_x_set[row_set2, 0] == 0):#make shorter
                if curve_x_set[row_set1, 0] != curve_x_get[row_get1, 0]:
                    curve_x_set[olig_num_2, 0] = curve_x_get[row_get1, 0]
                    curve_y_set[olig_num_2, 0] = curve_y_get[row_get1, 0]
                    olig_num_2 += 1
                    return 0, olig_num_2 - 1
                elif rows_get.size == 2:  
                    curve_x_set[olig_num_2, 0] = curve_x_get[row_get2, 0]
                    curve_y_set[olig_num_2, 0] = curve_y_get[row_get2, 0]
                    olig_num_2 += 1
                    return 0, olig_num_2 - 1
                else: return None, None


def action(curve_x_get, curve_y_get, curve_x_set, curve_y_set, x0, y0, what_fig, mode):
    global olig_num_2
    global tot_olig_num_2
    if olig_num_2 < tot_olig_num_2:
        x, y = find_olig(x0, y0, 10, what_fig)
        set_id, row = compare_curves(curve_x_get, curve_y_get, curve_x_set, curve_y_set, x, y, mode)
        if set_id == None:
            print(f'x={None} y={None} id={None}')
        else:
            if olig_num_2 != tot_olig_num_2:
                fig2_add_olig()
                plt.figure(fig2)
                plt.clf()
                plot_im(im_2_to_plot_2)
                plt.draw()
                #plt.gcf().canvas.flush_events()
                print(f'x={x} y={y} id={int(curve_x_set[row, 0])}')  
            else: 
                print(f'x={x} y={y} id={int(curve_x_set[row, 0])}')                 
    else:
        write_curve_to_file(path_out_curve, "t_curve_x_", "t_curve_y_", im_num + 1 - 1, curve_x_2, curve_y_2)
        olig_num_2 = 0
        if im_num < tot_im_num:
            next_image()

            add_all_olig()
            plt.figure(fig1)
            plot_im(im_1_to_plot)
            plt.draw()
            plt.gcf().canvas.flush_events()

            fig2_add_olig()
            plt.figure(fig2)
            plot_im(im_2_to_plot_2)
            plt.draw()
            plt.gcf().canvas.flush_events()
        else:
            plt.close('all')


def fig2_click(event):
    if event.button is MouseButton.MIDDLE:
        x0 = round(event.xdata)
        y0 = round(event.ydata)
        action(curve_x_2, curve_y_2, curve_x_2, curve_y_2, x0, y0, 2, 2)


def fig1_click(event):
    if event.button is MouseButton.MIDDLE:
        x0 = round(event.xdata)
        y0 = round(event.ydata)
        action(curve_x_1, curve_y_1, curve_x_2, curve_y_2, x0, y0, 1, 1)


def remove_click(event):
    if event.button is MouseButton.RIGHT:
        x0 = round(event.xdata)
        y0 = round(event.ydata)
        action(curve_x_2, curve_y_2, curve_x_2, curve_y_2, x0, y0, 2, 3)


x_max = 512
y_max = 512
tot_im_num = 7
im_num = 6

im_num -= 1 # numbering starts from 0
olig_num_2 = 0
data_field_1 = np.zeros(x_max * y_max)
data_field_2 = np.zeros(x_max * y_max)
copy_data_field_2 = np.zeros(x_max * y_max)
sum_data_field_2 = np.zeros(x_max * y_max)
new_id = 0
im_1 = np.zeros(x_max * y_max)
im_2 = np.zeros(x_max * y_max)
im_1_to_plot = np.zeros((x_max, y_max))
im_2_to_plot = np.zeros((x_max, y_max))
im_2_to_plot_2 = np.zeros((x_max, y_max))
path_inp_curve_1 = ".\\Data\\Output\\Curves\\" 
path_inp_curve_2 = ".\\Data\\Input\\Curves\\" 
path_out_curve = ".\\Data\\Output\\Curves\\"
path_inp_im = ".\\Data\\Input\\Raw_images\\"

curve_x_1, curve_y_1, tot_olig_num_1 = open_curve_from_file(path_inp_curve_1, path_inp_curve_2, "curve_x_", "curve_y_", im_num + 1, 1)
curve_x_2, curve_y_2, tot_olig_num_2 = open_curve_from_file(path_inp_curve_1, path_inp_curve_2, "curve_x_", "curve_y_", im_num + 2, 2)
im_1 = open_im(path_inp_im, "image_", im_num + 1)
im_2 = open_im(path_inp_im, "image_", im_num + 2)

im_num += 2

write_curve_to_file(path_out_curve, "t_curve_x_", "t_curve_y_", im_num - 1, curve_x_1, curve_y_1)
fig1 = plt.figure()
add_all_olig()
plot_im(im_1_to_plot)
plt.connect('button_press_event', fig1_click)
fig2 = plt.figure()
fig2_add_olig()
plot_im(im_2_to_plot_2)
plt.connect('button_press_event', fig2_click)
plt.connect('button_press_event', remove_click)
plt.show()



