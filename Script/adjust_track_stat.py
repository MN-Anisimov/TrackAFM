settings = gwy.gwy_app_settings_get()

# Resampling
for container in gwy.gwy_app_data_browser_get_containers():
	resampled_field = container['/0/data'].new_resampled(512, 512, INTERPOLATION_LINEAR)
	container['/0/data'] = resampled_field

# Remove scars
settings['/module/scars_remove/combine'] = False
settings['/module/scars_remove/combine_type'] = 0
settings['/module/scars_remove/max_width'] = 4
settings['/module/scars_remove/min_len'] = 16
settings['/module/scars_remove/threshold_high'] = 0.666
settings['/module/scars_remove/threshold_low'] = 0.25
settings['/module/scars_remove/type'] = 5
settings['/module/scars_remove/update'] = False
gwy.gwy_process_func_run('scars_remove', container, gwy.RUN_IMMEDIATE)
gwy.gwy_process_func_run('scars_remove', container, gwy.RUN_IMMEDIATE)
gwy.gwy_process_func_run('scars_remove', container, gwy.RUN_IMMEDIATE)	

# Align rows
settings['/module/align_rows/direction'] = 0
settings['/module/align_rows/do_extract'] = False
settings['/module/align_rows/do_plot'] = False
settings['/module/align_rows/masking'] = 2
settings['/module/align_rows/max_degree'] = 0
settings['/module/align_rows/method'] = 5
settings['/module/align_rows/trim_fraction'] = 0
gwy.gwy_process_func_run('align_rows', container, gwy.RUN_IMMEDIATE)

# Remove polynomial background
settings['/module/polylevel/col_degree'] = 3
settings['/module/polylevel/do_extract'] = False
settings['/module/polylevel/independent'] = 1
settings['/module/polylevel/masking'] = 2
settings['/module/polylevel/max_degree'] = 3
settings['/module/polylevel/row_degree'] = 3
settings['/module/polylevel/same_degree'] = True
gwy.gwy_process_func_run('polylevel', container, gwy.RUN_IMMEDIATE)

# Gaussian blurring
for container in gwy.gwy_app_data_browser_get_containers():	
	container['/0/data'].filter_gaussian(2)
	zmin, zmax = container['/0/data'].get_min_max()
	mask = container['/0/data'].duplicate()
	threshval = (abs(zmin) + 0.15 * zmax) / (zmax - zmin) * 100
	mask.grains_mark_height(mask, threshval, False)
	mask.grains_thin()
	gwy_app_data_browser_add_data_field(mask, container, True)
	
import numpy as np
import math
from math import sqrt

def is_start(i, j, type_field):
	current = i * x_max + j
	cvalue = type_field[current]
	if (cvalue == 2) or (cvalue == 3) or (cvalue == 4): # start, triple or cross
		return i, j
		
	
def what_type(i, j, data_field):
	lstep = i * x_max + j - 1
	rstep = i * x_max + j + 1
	bstep = (i + 1) * x_max + j
	tstep = (i - 1) * x_max + j
	current = i * x_max + j
	cvalue = data_field[current]
	lvalue = data_field[lstep]
	rvalue = data_field[rstep]
	bvalue = data_field[bstep]
	tvalue = data_field[tstep]
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
		
		
def what_rank(i, j, data_field):
	lstep = i * x_max + j - 1
	rstep = i * x_max + j + 1
	bstep = (i + 1) * x_max + j
	tstep = (i - 1) * x_max + j
	current = i * x_max + j
	cvalue = data_field[current]
	lvalue = data_field[lstep]
	rvalue = data_field[rstep]
	bvalue = data_field[bstep]
	tvalue = data_field[tstep]
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


def set_type_field(data_field):	
	type_field = data_field.duplicate()
	for i in range(1, y_max - 1):
		for j in range(1, x_max - 1):
			if what_type(i, j, data_field) != None:
				type_field[i * x_max + j] = what_type(i, j, data_field)
				
	for i in range(1, y_max - 1):
		for j in range(1, x_max - 1):
			if find_triple_diag(i, j, type_field) != None:
				type_field[i * x_max + j] = find_triple_diag(i, j, type_field)
	return type_field
	
	
def set_rank_field(data_field, type_field):	
	rank_field = data_field.duplicate()
	for i in range(1, y_max - 1):
		for j in range(1, x_max - 1):
			if what_rank(i, j, data_field) != None:
				rank_field[i * x_max + j] = what_rank(i, j, data_field)
				
	for i in range(1, y_max - 1):
		for j in range(1, x_max - 1):
			if type_field[i * x_max + j] == 5:
				rank_field[i * x_max + j] = 1
				
	return rank_field


def move(i, j, data_field, type_field, id_field):
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
			
							
def find_orient(length, curve_x, curve_y, min_angle, max_angle, orient_field):
	orient_field = orient_field
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
		

def radius(curve_x, curve_y, window, blured_field, curve_field, oriented_curve):
	curve_field = curve_field
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
		

def density(data_field):
	density = 0
	for i in range(1, y_max - 1):
		for j in range(1, x_max - 1):
			if data_field[i * x_max + j] == 1:
				density = density + 1
			else:
				pass
	density = density / float((x_max - 1) * (y_max - 1))
	
	return density


for container in gwy.gwy_app_data_browser_get_containers():	
	data_field = container['/4/data']
	blured_field = container['/0/data']

x_max = data_field.get_xres()
y_max = data_field.get_yres()	
start = np.zeros([1, 2])
curve_x = np.zeros((1, 2000))
curve_y = np.zeros((1, 2000))
type_field = data_field.duplicate() # 1 - normal, 2 - start, 3 - triple, 4 cross, 5 - triple with diag neigh
id_field = data_field.duplicate()
rank_field = data_field.duplicate()
curve_field = data_field.duplicate()
orient_field = data_field.duplicate()
curve_id = 0

id_field.multiply(0)
curve_field.multiply(0)
orient_field.multiply(0)
type_field = set_type_field(data_field)
rank_field = set_rank_field(data_field, type_field)
start = find_start(type_field)
stop_list = np.array([0])

while True:
	if start.shape[0] == stop_list.shape[0]:
		break
	else:
		for n in range(1, start.shape[0]):
			if any(stop_list == n):
				pass
			else:
				curve_id = curve_id + 1
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
					
						(di, dj) = move(temp_i, temp_j, data_field, type_field, id_field)
						if (di == 0) and (dj == 0):
							break
						else:
							last_elem = np.max(np.nonzero(curve_x[curve_id, :])) + 1
							temp_i = temp_i + di
							temp_j = temp_j + dj
							temp_index = temp_i * x_max + temp_j
							id_field[temp_index] = curve_id
							rank_field[temp_index] = rank_field[temp_index] - 1 
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

# Calculate density
d = density(data_field) 

# Delete short curves
to_remove = np.array([0])
for i in range(1, curve_x.shape[0]):
	last_elem = np.max(np.nonzero(curve_x[i, :]))
	if last_elem < 29:
		to_remove = np.append(to_remove, np.array([i]), axis = 0)

curve_x = np.delete(curve_x, to_remove, 0)
curve_y = np.delete(curve_y, to_remove, 0)

# Find orientations
(oriented_curve, orient_field) = find_orient(15, curve_x, curve_y, 0 , math.pi / 12, orient_field)

# Measure radii of curvature
(radii, curvat, curve_field, heights_orient, radii_orient, curvat_orient) = radius(curve_x, curve_y, 30, blured_field, curve_field, oriented_curve)
(radii_hyst, radii_bin) = np.histogram(radii, bins=20, range = (0, 1000))
(curvat_hyst, curvat_bin) = np.histogram(curvat, bins=20, range = (0, 0.03))
(heights_hyst, heights_bin) = np.histogram(heights_orient, bins=20, range = (0, 30))

#print(d)
#print(curve_x.shape[0])
#print(list(radii_hyst))
#print(list(radii_bin))
print(list(curvat_hyst))
print(list(curvat_bin))
print(list(heights_hyst))
print(list(heights_bin))
for i in range(heights_orient.shape[0]):
	print heights_orient[i], curvat_orient[i]

for i in range(y_max):			
	for j in range(x_max):
		if (i == 0) or (i == y_max - 2) or (j == x_max - 2) or (j == x_max - 2):
			data_field[i * x_max + j] = 0
		else:
			data_field[i * x_max + j] = orient_field[i * x_max + j] #+ curve_field[i * x_max + j]
				
data_field.data_changed()
