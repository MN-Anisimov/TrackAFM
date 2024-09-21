import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import codecs
from matplotlib.backend_bases import MouseButton
from numpy.ma import masked_array


def find_olig(x0, y0, w):
    if x0 % x_max - 1 < w:
        x1 = x0 - x0 % x_max + 1
    else:
        x1 = x0 - w

    if x_max - x0 % x_max < w:
        x2 = x0 + x_max - x0 % x_max 
    else:
        x2 = x0 + w

    if y0 % y_max - 1 < w:
        y1 = y0 - y0 % y_max + 1
    else:
        y1 = y0 - w

    if y_max - y0 % y_max < w:
        y2 = y0 + y_max - y0 % y_max 
    else:
        y2 = y0 + w
    
    list_x = [_ + y0 * x_max + x1 for _ in range(x2 - x1 + 1)]
    list_y = [y1 * x_max + x0 + x_max * _ for _ in range(y2 - y1 + 1)]

    if np.any(data_field_1[list_x]) or np.any(data_field_1[list_y]):
        if np.any(data_field_1[list_x]):
            index = list_x[int(np.min(np.nonzero(data_field_1[list_x])))]
        else:
            index = list_y[int(np.min(np.nonzero(data_field_1[list_y])))]
        return index % x_max, index // x_max
    else:
        return None, None


def next_image():
    global im_2
    global im_num
    global curve_x_1
    global curve_y_1
    global curve_x_2
    global curve_y_2

    curve_x_1 = curve_x_2
    curve_y_1 = curve_y_2

    with codecs.open('curve_x_' + str(tot_im_num)  + '.txt', encoding='utf-8-sig') as f:
        curve_x_2 = np.loadtxt(f)
    with codecs.open('curve_y_' + str(tot_im_num)  + '.txt', encoding='utf-8-sig') as f:
        curve_y_2 = np.loadtxt(f)
    with codecs.open('im_' + str(tot_im_num)  + '.txt', encoding='utf-8-sig') as f:
        im_2 = np.loadtxt(f)

    im_num += 1
    

def plot_im(im):
     plt.matshow(masked_array(im, im > 0.5), cmap=matplotlib.cm.Greys_r, fignum = 0)
     plt.matshow(masked_array(im, im < 0.5), cmap=ListedColormap(["red"]), fignum = 0)


def add_olig(mode):
    global olig_num_2
    global data_field_1
    global data_field_2
    global im_to_plot_1
    global im_to_plot_2
    if mode == 1:
        for i in range(tot_olig_num_1):
            indexes = curve_y_1[i, np.nonzero(curve_y_1[i, 1:])[0] + 1] * x_max + curve_x_1[i, np.nonzero(curve_x_1[i, 1:])[0] + 1]
            np.put(data_field_1, list(indexes), 1) # data_field must be 1D array
            im_to_plot_1 = im_1 + data_field_1.reshape((x_max, y_max))
            im_to_plot_1[im_to_plot_1 > 0.5] = 1
    else:
        indexes = curve_y_2[olig_num_2, np.nonzero(curve_y_2[olig_num_2, 1:])[0] + 1] * x_max + curve_x_2[olig_num_2, np.nonzero(curve_x_2[olig_num_2, 1:])[0] + 1]
        data_field_2 = data_field_2 * 0
        np.put(data_field_2, list(indexes), 1) # data_field must be 1D array
        im_to_plot_2 = im_2 + data_field_2.reshape((x_max, y_max))
        im_to_plot_2[im_to_plot_2 > 0.5] = 1
        olig_num_2 += 1


#def set_click(event):    
#    if event.button is MouseButton.MIDDLE:
#        data_field
#        plt.matshow(data_field, fignum = 0)
#        plt.draw()
#        plt.gcf().canvas.flush_events()
#        print(f'x={x} y={y} value={value}')
        
        
def get_click(event):
    global olig_num_2
    global im_num
    if event.button is MouseButton.MIDDLE:
        if olig_num_2 < tot_olig_num_2:
            x0 = round(event.xdata)
            y0 = round(event.ydata)
            x, y = find_olig(x0, y0, 10)
            row = np.where((curve_x_1[:, 1:] == x) & (curve_y_1[:, 1:] == y))[0]
            if row.size > 0:
                curve_x_2[olig_num_2, 0] = curve_x_1[row, 0]
                curve_y_2[olig_num_2, 0] = curve_y_1[row, 0]
                add_olig(2)
                plt.figure(fig2)
                plot_im(im_to_plot_2)
                plt.draw()
                plt.gcf().canvas.flush_events()
            print(f'x={x} y={y} id={curve_x_1[row, 0]}')
        else:
            np.savetxt('t_curve_x_' + str(im_num) + '.txt', curve_x_2, '%d')
            np.savetxt('t_curve_y_' + str(im_num) + '.txt', curve_y_2, '%d')

            olig_num_2 = -1
            if im_num < tot_im_num:
                next_image()

                add_olig(1)
                plt.figure(fig1)
                plot_im(im_to_plot_1)
                plt.draw()
                plt.gcf().canvas.flush_events()

                add_olig(2)
                plt.figure(fig2)
                plot_im(im_to_plot_2)
                plt.draw()
                plt.gcf().canvas.flush_events()
            else:
                plt.close('all')

x_max = 512
y_max = 512
tot_im_num = 3
im_num = 0
olig_num_2 = -1
data_field_1 = np.zeros(x_max * y_max)
data_field_2 = np.zeros(x_max * y_max)
im_to_plot_1 = np.zeros(x_max * y_max)
im_to_plot_2 = np.zeros(x_max * y_max)
im_1 = np.zeros((x_max, y_max))
im_2 = np.zeros((x_max, y_max))

with codecs.open('curve_x_1.txt', encoding='utf-8-sig') as f:
    curve_x_1 = np.loadtxt(f)
with codecs.open('curve_y_1.txt', encoding='utf-8-sig') as f:
    curve_y_1 = np.loadtxt(f)
with codecs.open('curve_x_2.txt', encoding='utf-8-sig') as f:
    curve_x_2 = np.loadtxt(f)
with codecs.open('curve_y_2.txt', encoding='utf-8-sig') as f:
    curve_y_2 = np.loadtxt(f)
with codecs.open('im_1.txt', encoding='utf-8-sig') as f:
    im_1 = np.loadtxt(f)
with codecs.open('im_2.txt', encoding='utf-8-sig') as f:
    im_2 = np.loadtxt(f)

im_num = 2

tot_olig_num_1 = curve_x_1.shape[0]
tot_olig_num_2 = curve_x_2.shape[0]

curve_x_1 = np.hstack((np.array(range(1, tot_olig_num_1 + 1)).reshape(tot_olig_num_1, 1), curve_x_1))
curve_y_1 = np.hstack((np.array(range(1, tot_olig_num_1 + 1)).reshape(tot_olig_num_1, 1), curve_y_1))
curve_x_2 = np.hstack((np.array(range(1, tot_olig_num_2 + 1)).reshape(tot_olig_num_2, 1), curve_x_2))
curve_y_2 = np.hstack((np.array(range(1, tot_olig_num_2 + 1)).reshape(tot_olig_num_2, 1), curve_y_2))

fig1 = plt.figure()
add_olig(1)
plot_im(im_to_plot_1)
plt.connect('button_press_event', get_click)

fig2 = plt.figure()
add_olig(2)
plot_im(im_to_plot_2)
plt.connect('button_press_event', get_click)

plt.show()



