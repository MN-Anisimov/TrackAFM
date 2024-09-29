import matplotlib.pyplot as plt
import numpy as np
import codecs
import math
import copy
from math import sqrt
import cv2


def open_curves(path, num):
    with codecs.open(path + "\\curve_x_" + str(num) + ".txt", encoding='utf-8-sig') as f:
        curve_x = np.loadtxt(f)
    with codecs.open(path + "\\curve_y_" + str(num) + ".txt", encoding='utf-8-sig') as f:
        curve_y = np.loadtxt(f)
    if len(curve_x.shape) == 1:
        curve_x = curve_x.reshape(1, curve_x.shape[0])
        curve_y = curve_y.reshape(1, curve_y.shape[0])
    else:
        pass

    return curve_x, curve_y


def open_im(path, num):
	with codecs.open(path_input_im + "\\image_" + str(num) + ".txt", encoding='utf-8-sig') as f:
		field = np.loadtxt(f)
	field = field.ravel()

	return field


def data_field_from_curves(curve_x, curve_y):
	field = np.zeros((y_max, x_max))
	field = field.ravel()
	for i in range(curve_x.shape[0]):
		j_max = np.max(np.nonzero(curve_x[i, :]))
		for j in range(j_max + 1):
			field[int(curve_y[i, j - 1]) * x_max + int(curve_x[i, j - 1])] = 1

	return field


def find_orient(length, curve_x, curve_y, min_angle, max_angle, orient_field_arg):
	orient_field = copy.deepcopy(orient_field_arg)
	oriented_curve = np.zeros((curve_x.shape[0], curve_x.shape[1])) + 5
	for i in range(curve_x.shape[0]):
		j_max = np.max(np.nonzero(curve_x[i, :]))
		for j in range(j_max + 1):
			if (j + 1) % length == 0:
				x = curve_x[i, j] - curve_x[i, j - length + 1]
				y = curve_y[i, j] - curve_y[i, j - length + 1]
				if (x*0 + y*1) / (sqrt(x*x + y*y)) > 1:
					alpha = math.acos(1)
				else:
					if (x*0 + y*1) / (sqrt(x*x + y*y)) < -1:
						alpha = math.acos(-1)
					else:
						alpha = math.acos((x*0 + y*1) / (sqrt(x*x + y*y)))
				for jj in range(length):
					if (abs(alpha) >= min_angle and abs(alpha) <= max_angle) or (abs(alpha) >= math.pi - max_angle and abs(alpha) <= math.pi - min_angle):
						oriented_curve[i, j - length + jj + 1] = alpha
						orient_field[int(curve_y[i, j - length + jj + 1]) * x_max + int(curve_x[i, j - length + jj + 1])] = 1
			else:
				pass
	
	return oriented_curve, orient_field


def height_at_angle(curve_x, curve_y, oriented_curve_arg, blured_field, orient_field_arg):
	oriented_curve = copy.deepcopy(oriented_curve_arg)
	orient_field = copy.deepcopy(orient_field_arg)
	heights_orient = np.array([])
	for i in range(curve_x.shape[0]):
		j_max = np.max(np.nonzero(curve_x[i, :]))
		for j in range(j_max + 1):
			if oriented_curve[i, j] != 5:
				height_value = blured_field[int(curve_y[i, j]) * x_max + int(curve_x[i, j])] * 1e+9
				heights_orient = np.append(heights_orient, ([height_value]), axis = 0)

	return heights_orient


def heights_0_pi(curve_x, curve_y, num_of_angles):
	global orient_field
	heights = np.zeros((num_of_angles, 3))
	dangle = math.pi / num_of_angles
	for i in range(num_of_angles):
		oriented_curve, orient_field = find_orient(15, curve_x, curve_y, dangle * i, dangle * (i + 1), orient_field)
		heights_1 = height_at_angle(curve_x, curve_y, oriented_curve, blured_field, orient_field)
		heights[i, 0] = dangle * (i + 1)
		heights[i, 1] = np.average(heights_1)
		heights[i, 2] = np.std(heights_1)
	return heights


def radius(curve_x, curve_y, window, blured_field, curve_field_arg, oriented_curve):
	curve_field = copy.deepcopy(curve_field_arg)
	radii = np.array([])
	curvat = np.array([])
	heights_orient = np.array([])
	radii_orient = np.array([])
	curvat_orient = np.array([])
	for i in range(curve_x.shape[0]):
		j_max = np.max(np.nonzero(curve_x[i, :]))
		for j in range(j_max + 1):
			if j >= window - 1:
				x1 = curve_x[i, j] - curve_x[i, j - window // 2]
				y1 = curve_y[i, j] - curve_y[i, j - window // 2]
				x2 = curve_x[i, j - window // 2] - curve_x[i, j - window + 1]
				y2 = curve_y[i, j - window // 2] - curve_y[i, j - window + 1]
				x3 = curve_x[i, j] - curve_x[i, j - window + 1]
				y3 = curve_y[i, j] - curve_y[i, j - window + 1]
				if (x1*x2 + y1*y2) / (sqrt(x1*x1 + y1*y1)*sqrt(x2*x2 + y2*y2)) > 1:
					alpha = math.acos(1)
				else:
					if (x1*x2 + y1*y2) / (sqrt(x1*x1 + y1*y1)*sqrt(x2*x2 + y2*y2)) < -1:
						alpha = math.acos(-1)
					else:
						alpha = math.acos((x1*x2 + y1*y2) / (sqrt(x1*x1 + y1*y1)*sqrt(x2*x2 + y2*y2)))
				if alpha != 0:
					radius = sqrt(x3*x3 + y3*y3) / (2 * math.sin(alpha)) * 5000 / 512
					radii = np.append(radii, ([radius]), axis = 0)
					curvat = np.append(curvat, ([1/radius]), axis = 0)
					if oriented_curve[i, j - window // 2] != 5:
						height_value = blured_field[int(curve_y[i, j]) * x_max + int(curve_x[i, j])] * 1e+9
						heights_orient = np.append(heights_orient, ([height_value]), axis = 0)
						radii_orient = np.append(radii_orient, ([radius]), axis = 0)
						curvat_orient = np.append(curvat_orient, ([1/radius]), axis = 0)
					else:
						pass
					curve_field[int(curve_y[i, j - 1]) * x_max + int(curve_x[i, j - 1])] = radius
				else:
					pass
			else:
				pass
				
	return radii, curvat, curve_field, heights_orient, radii_orient, curvat_orient


def save_plot(path, name, num, x, y):
	np.savetxt(path + name + str(num) + ".txt", np.c_[x, y], '%.5f', encoding = 'utf-8-sig')


def density(field):
	density = 0
	for i in range(1, y_max - 1):
		for j in range(1, x_max - 1):
			if field[i * x_max + j] == 1:
				density = density + 1
			else:
				pass
	density = density / float((x_max - 1) * (y_max - 1))
	
	return density


def plot_result(field):
	field = field.reshape((x_max, y_max))		
	plt.matshow(field)
	plt.show()


path_input = ".\\Data\\Output\\Curves"
path_input_im = ".\\Data\\Input\\Raw_images"
path_output_plots = ".\\Data\\Output\\Plots\\"
y_max, x_max = 512, 512
num = 10
curve_x, curve_y = open_curves(path_input, num)
data_field = data_field_from_curves(curve_x, curve_y)
blured_field = open_im(path_input_im, num)
orient_field = copy.deepcopy(data_field)
curve_field = copy.deepcopy(data_field)
orient_field = 0 * orient_field
curve_field = 0 * curve_field


heights = heights_0_pi(curve_x, curve_y, 12)
plt.xlabel('Angle, rad')
plt.ylabel('Average heights, nm')
plt.errorbar(heights[:, 0], heights[:, 1], yerr = heights[:, 2], linestyle='None', marker='o')
plt.show()
#oriented_curve, orient_field = find_orient(15, curve_x, curve_y, 0, math.pi / 20, orient_field)
#heights = height_at_angle(curve_x, curve_y, oriented_curve, blured_field, orient_field)
#plot_result(orient_field)

#d = density(data_field) 
#oriented_curve, orient_field = find_orient_abs(15, curve_x, curve_y, 0 , math.pi / 12, orient_field)
#radii, curvat, curve_field, heights_orient, radii_orient, curvat_orient = radius(curve_x, curve_y, 30, blured_field, curve_field, oriented_curve)
#radii_hyst, radii_bin = np.histogram(radii, bins=20, range = (0, 1000))
#curvat_hyst, curvat_bin = np.histogram(curvat, bins=20, range = (0, 0.03))
#heights_hyst, heights_bin = np.histogram(heights_orient, bins=20, range = (0, 30))
#save_plot(path_output_plots, "curvet_hyst_", num, list(curvat_bin[1 : ]), list(curvat_hyst))

