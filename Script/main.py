import matplotlib.pyplot as plt
import numpy as np
import codecs
import math
import copy
from math import sqrt
import cv2


def is_start(i, j, type_field):
	current = i * x_max + j
	cvalue = type_field[current]
	if (cvalue == 2) or (cvalue == 3) or (cvalue == 4): # start, triple or cross
		return i, j

	
def what_type(i, j, mask_field):
	lstep = i * x_max + j - 1
	rstep = i * x_max + j + 1
	bstep = (i + 1) * x_max + j
	tstep = (i - 1) * x_max + j
	current = i * x_max + j
	cvalue = mask_field[current]
	lvalue = mask_field[lstep]
	rvalue = mask_field[rstep]
	bvalue = mask_field[bstep]
	tvalue = mask_field[tstep]
	if lvalue != 0:
		lvalue = 1
	if rvalue != 0:
		rvalue = 1
	if bvalue != 0:
		bvalue = 1
	if tvalue != 0:
		tvalue = 1
	
	s = lvalue + rvalue + bvalue + tvalue
	
	if (cvalue == 1) and (s == 1): # start
		return float(2)
	if (cvalue == 1) and (s == 3): # triple
		return float(3)
	if (cvalue == 1) and (s == 4): # cross
		return float(4)
		
		
def what_rank(i, j, mask_field):
	lstep = i * x_max + j - 1
	rstep = i * x_max + j + 1
	bstep = (i + 1) * x_max + j
	tstep = (i - 1) * x_max + j
	current = i * x_max + j
	cvalue = mask_field[current]
	lvalue = mask_field[lstep]
	rvalue = mask_field[rstep]
	bvalue = mask_field[bstep]
	tvalue = mask_field[tstep]
	if lvalue != 0:
		lvalue = 1
	if rvalue != 0:
		rvalue = 1
	if bvalue != 0:
		bvalue = 1
	if tvalue != 0:
		tvalue = 1
	
	s = lvalue + rvalue + bvalue + tvalue
	
	if (cvalue == 1) and (s == 1): # start
		return float(1)
	if (cvalue == 1) and (s == 3): # triple
		return float(3)
	if (cvalue == 1) and (s == 4): # cross
		return float(4)


def find_triple_diag(i, j, type_field):
	blstep = (i + 1) * x_max + j - 1
	brstep = (i + 1) * x_max + j + 1
	tlstep = (i - 1) * x_max + j - 1
	trstep = (i - 1) * x_max + j + 1
	current = i * x_max + j
	cvalue = type_field[current]
	blvalue = type_field[blstep]
	brvalue = type_field[brstep]
	tlvalue = type_field[tlstep]
	trvalue = type_field[trstep]
	
	lstep = i * x_max + j - 1
	rstep = i * x_max + j + 1
	bstep = (i + 1) * x_max + j
	tstep = (i - 1) * x_max + j
	current = i * x_max + j
	cvalue = type_field[current]
	lvalue = type_field[lstep]
	rvalue = type_field[rstep]
	bvalue = type_field[bstep]
	tvalue = type_field[tstep]
	
	s_diag = 0
	s_neigh = 0
	
	if blvalue == 3 or blvalue == 5:
		s_diag = s_diag + 1
	if brvalue == 3 or brvalue == 5:
		s_diag = s_diag + 1
	if tlvalue == 3 or tlvalue == 5:
		s_diag = s_diag + 1
	if trvalue == 3 or trvalue == 5:
		s_diag = s_diag + 1
		
	if lvalue == 3 or lvalue == 5:
		s_neigh = s_neigh + 1
	if rvalue == 3 or rvalue == 5:
		s_neigh = s_neigh + 1
	if bvalue == 3 or bvalue == 5:
		s_neigh = s_neigh + 1
	if tvalue == 3 or tvalue == 5:
		s_neigh = s_neigh + 1
	
	if (cvalue == 3 or cvalue == 5) and (s_diag != 0 or s_neigh != 0): 
		return float(5)


def find_start(type_field):
	start = np.array([[1, 2]])
	for i in range(1, y_max - 1):
		for j in range(1, x_max - 1):
			if is_start(i, j, type_field) != None:
				(start_y, start_x) = is_start(i, j, type_field)
				start = np.append(start, np.array([[start_y, start_x]]), axis = 0)

	return start


def set_type_field(mask_field):	
	type_field = copy.deepcopy(mask_field)
	for i in range(1, y_max - 1):
		for j in range(1, x_max - 1):
			if what_type(i, j, mask_field) != None:
				type_field[i * x_max + j] = what_type(i, j, mask_field)
				
	for i in range(1, y_max - 1):
		for j in range(1, x_max - 1):
			if find_triple_diag(i, j, type_field) != None:
				type_field[i * x_max + j] = find_triple_diag(i, j, type_field)

	return type_field
	
	
def set_rank_field(mask_field, type_field):	
	rank_field = copy.deepcopy(mask_field)
	for i in range(1, y_max - 1):
		for j in range(1, x_max - 1):
			if what_rank(i, j, mask_field) != None:
				rank_field[i * x_max + j] = what_rank(i, j, mask_field)
				
	for i in range(1, y_max - 1):
		for j in range(1, x_max - 1):
			if type_field[i * x_max + j] == 5:
				rank_field[i * x_max + j] = 1
				
	return rank_field


def move(i, j, type_field, id_field, rank_field):
	di = 0
	dj = 0
	for ii in range(3):
		for jj in range(3):
			if (ii * jj != 1) and (i != 0) and (i != y_max - 1) and (j != 0) and (j != x_max - 1):
				run_index = (i + (ii - 1)) * x_max + j + (jj - 1)
				index = i * x_max + j
				is_neigh = (type_field[run_index] != 5 and type_field[run_index] != 3 and type_field[run_index] != 4) and (rank_field[run_index] != 0) and ((ii - 1) * (jj - 1) == 0)			
				if is_neigh:
					return ii - 1, jj - 1
				else:
					continue
	
	for ii in range(3):
		for jj in range(3):
			if (ii * jj != 1) and (i != 0) and (i != y_max - 1) and (j != 0) and (j != x_max - 1):
				run_index = (i + (ii - 1)) * x_max + j + (jj - 1)
				index = i * x_max + j
				is_neigh_5 = (type_field[run_index] == 5) and (rank_field[run_index] != 0) and ((ii - 1) * (jj - 1) == 0)			
				if is_neigh_5:
					return ii - 1, jj - 1
				else:
					continue			
		
	for ii in range(3):
		for jj in range(3):
			if (ii * jj != 1) and (i != 0) and (i != y_max - 1) and (j != 0) and (j != x_max - 1):
				run_index = (i + (ii - 1)) * x_max + j + (jj - 1)
				index = i * x_max + j
				is_neigh_34 = (type_field[run_index] == 3 or type_field[run_index] == 4) and (rank_field[run_index] != 0) and ((ii - 1) * (jj - 1) == 0)
				if is_neigh_34:
					return ii - 1, jj - 1
				else:
					continue
					
	return 0, 0				
		

def track(mode):
	global start
	global stop_list
	global curve_id
	global curve_x, curve_y
	global mask_field, type_field, id_field, rank_field
	curve_x = np.zeros((1, 2000))
	curve_y = np.zeros((1, 2000))
	id_field = 0 * id_field
	type_field = set_type_field(mask_field)
	rank_field = set_rank_field(mask_field, type_field)
	start = find_start(type_field)
	stop_list = np.array([0])
	curve_id = 0

	while True:
		if start.shape[0] == stop_list.shape[0]:
			break
		else:
			for n in range(1, start.shape[0]):
				if any(stop_list == n):
					pass
				else:
					curve_id += 1
					last_elem = 0
					temp_i = start[n, 0]
					temp_j = start[n, 1]
					temp_index = temp_i * x_max + temp_j
					id_field[temp_index] = curve_id
					rank_field[temp_index] = rank_field[temp_index] - 1 
					curve_x = np.append(curve_x, np.zeros((1, 2000)), axis = 0)
					curve_y = np.append(curve_y, np.zeros((1, 2000)), axis = 0)
					curve_x[curve_id, last_elem] = temp_j
					curve_y[curve_id, last_elem] = temp_i
					if rank_field[temp_index] == 0:
						stop_list = np.append(stop_list, ([n]), axis = 0)
					else:
						pass
				
					while True:
						if temp_i == 0 or temp_i == y_max - 1 or temp_j == 0 or temp_j == x_max - 1:
							break
						else:	
							(di, dj) = move(temp_i, temp_j, type_field, id_field, rank_field)
							if (di == 0) and (dj == 0):
								break
							else:
								temp_i = temp_i + di
								temp_j = temp_j + dj
								last_elem = np.max(np.nonzero(curve_x[curve_id, :])) + 1
								temp_index = temp_i * x_max + temp_j
								id_field[temp_index] = curve_id
								rank_field[temp_index] = rank_field[temp_index] - 1
								if (temp_i != 0) & (temp_j != 0) & ((temp_i - y_max + 1) != 0) & ((temp_j - x_max + 1) != 0):
									curve_x[curve_id, last_elem] = temp_j
									curve_y[curve_id, last_elem] = temp_i
								is_end = (type_field[temp_index] == 2) or (type_field[temp_index] == 3) or (type_field[temp_index] == 4)
								if is_end and rank_field[temp_index] == 0:
									stop_list_value = np.where((start[:, 0] == temp_i) & (start[:, 1] == temp_j))[0]
									stop_list = np.append(stop_list, stop_list_value, axis = 0)
									break
								else:
									if is_end:
										break
									else:
										pass

					if mode == 2:
						# dublicate curves with at least one branching pix at the ends
						index_left = int(curve_y[curve_id, 0]) * x_max + int(curve_x[curve_id, 0])			
						index_right = int(curve_y[curve_id, last_elem]) * x_max + int(curve_x[curve_id, last_elem])
						if (type_field[index_left] == 3) or (type_field[index_right] == 3):
							curve_x = np.insert(curve_x, curve_id, curve_x[curve_id, :], axis = 0)
							curve_y = np.insert(curve_y, curve_id, curve_y[curve_id, :], axis = 0)
							curve_id += 1
						else:
							pass
					else:
						pass
									
	curve_x = np.delete(curve_x, np.array([0]), 0)
	curve_y = np.delete(curve_y, np.array([0]), 0)

	return curve_x, curve_y							
				

def remove_bad_curves(height):
	global curve_x, curve_y
	global height_field
	to_remove = np.array([])
	for i in range(1, curve_x.shape[0]):
		av_height = 0
		j_max = np.max(np.nonzero(curve_x[i, :]))
		for j in range(j_max + 1):
			av_height = av_height + height_field[int(curve_y[i, j - 1]) * x_max + int(curve_x[i, j - 1])] 
		av_height = av_height / (j_max + 1)
		if av_height < height:
			to_remove = np.append(to_remove, np.array([i]), axis = 0)
		else:
			pass

	to_remove = np.array(to_remove, dtype = np.int64)
	curve_x = np.delete(curve_x, to_remove, 0)
	curve_y = np.delete(curve_y, to_remove, 0)

	return curve_x, curve_y


def update_mask(curves_x, curves_y):
	global mask_field
	mask_field = mask_field * 0
	for i in range(curve_x.shape[0]):
		j_max = np.max(np.nonzero(curve_x[i, :]))
		for j in range(j_max + 1):
			mask_field[int(curve_y[i, j]) * x_max + int(curve_x[i, j])] = 1

	return mask_field


def track_and_update(height):
	global curves_x, curves_y
	curve_x, curve_y = track(1)
	curve_x, curve_y = remove_bad_curves(height)
	update_mask(curve_x, curve_y)
	curve_x, curve_y = track(2)


def open_mask_im(path_im, path_mask, num):
	with codecs.open(path_im + "\\image_" + str(num) + ".txt", encoding='utf-8-sig') as f:
		im = np.loadtxt(f)
	im = im.ravel()

	field = cv2.imread(path_mask + "\\mask_" + str(num) + ".tiff")
	field = np.sum(field, axis = 2)
	field[field > 0] = 1
	(y_max, x_max) = field.shape
	field = field.ravel()

	return field, im, y_max, x_max


def save_curves(path_out_curve, num, curve_x, curve_y):
	np.savetxt(path_out_curve + "\\curve_x_" + str(num) + ".txt", curve_x, '%d', encoding = 'utf-8-sig')
	np.savetxt(path_out_curve + "\\curve_y_" + str(num) + ".txt", curve_y, '%d', encoding = 'utf-8-sig')


def show_mask():
	global mask_field				
	plt.matshow(mask_field.reshape((x_max, y_max)))
	plt.show()


path_input_mask = ".\\Data\\Input\\Masks"
path_input_im = ".\\Data\\Input\\Raw_images"
path_output = ".\\Data\\Output\\Curves"
num = 8

mask_field, height_field, y_max, x_max = open_mask_im(path_input_im, path_input_mask, num)
start = np.zeros([1, 2])
curve_x = np.zeros((1, 2000))
curve_y = np.zeros((1, 2000))
type_field = copy.deepcopy(mask_field) # 1 - normal, 2 - start, 3 - triple, 4 cross, 5 - triple with diag neigh
id_field = copy.deepcopy(mask_field)
rank_field = copy.deepcopy(mask_field)

track_and_update(4e-9)
save_curves(path_output, num, curve_x, curve_y)
show_mask()
curve_x


