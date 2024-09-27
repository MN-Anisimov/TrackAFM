import codecs
import numpy as np
import copy


def open_tracked_oligo(filename_x, filename_y):
    with codecs.open(filename_x, encoding='utf-8-sig') as f1:
        olig_x = np.loadtxt(f1)
    with codecs.open(filename_y, encoding='utf-8-sig') as f2:
        olig_y = np.loadtxt(f2)
    
    return olig_x, olig_y


def remove_unnecessary(curve_x, curve_y, init_matchings):
    double_sided = list(np.where(np.sum(init_matchings, axis=1) == 2)[0])
    curve_x = np.delete(curve_x, double_sided, 0)
    curve_y = np.delete(curve_y, double_sided, 0)
    flip_curve_x = copy.deepcopy(curve_x)
    flip_curve_y = copy.deepcopy(curve_y)
    for i in range(curve_x.shape[0]):
        flip_curve_x[i, 1 : np.max(np.nonzero(curve_x[i, :])) + 1] = np.flip(curve_x[i, 1 : np.max(np.nonzero(curve_x[i, :])) + 1])
        flip_curve_y[i, 1 : np.max(np.nonzero(curve_y[i, :])) + 1] = np.flip(curve_y[i, 1 : np.max(np.nonzero(curve_y[i, :])) + 1])
    sum = curve_x + flip_curve_x + curve_y + flip_curve_y
    unique_rows = list(np.unique(sum[:, 1:], return_index = True, axis = 0)[1])   
    curve_x, curve_y = curve_x[unique_rows, :], curve_y[unique_rows, :]    
    curve_x, curve_y = curve_x[curve_x[:,0].argsort()], curve_y[curve_y[:,0].argsort()]

    return curve_x, curve_y


def oligs_to_match(olig_x, olig_y, init_olig_x, init_olig_y):
    matchings = np.zeros((olig_x.shape[0], olig_x.shape[0]))
    for i in range(olig_x.shape[0]):
        end_index_1 = np.max(np.nonzero(olig_x[i, :]))
        for j in range(init_olig_x.shape[0]):
            if i != j:
                end_index_2 = np.max(np.nonzero(init_olig_x[j, :]))
               
                s_x1 = olig_x[i, 1]              
                s_y1 = olig_y[i, 1]
                e_x1 = olig_x[i, end_index_1]
                e_y1 = olig_y[i, end_index_1]
                s_x2 = init_olig_x[j, 1]
                s_y2 = init_olig_y[j, 1]
                e_x2 = init_olig_x[j, end_index_2]
                e_y2 = init_olig_y[j, end_index_2]

                s_x1_next = olig_x[i, 2]
                s_y1_next = olig_y[i, 2]
                e_x1_prev = olig_x[i, end_index_1 - 1]
                e_y1_prev = olig_y[i, end_index_1 - 1]
                s_x2_next = init_olig_x[j, 2]
                s_y2_next = init_olig_y[j, 2]
                e_x2_prev = init_olig_x[j, end_index_2 - 1]
                e_y2_prev = init_olig_y[j, end_index_2 - 1]

                ss = (s_x1 == s_x2) and (s_y1 == s_y2) and (s_x1_next != s_x2_next or s_y1_next != s_y2_next)
                se = (s_x1 == e_x2) and (s_y1 == e_y2) and (s_x1_next != e_x2_prev or s_y1_next != e_y2_prev)
                es = (e_x1 == s_x2) and (e_y1 == s_y2) and (e_x1_prev != s_x2_next or e_y1_prev != s_y2_next)
                ee = (e_x1 == e_x2) and (e_y1 == e_y2) and (e_x1_prev != e_x2_prev or e_y1_prev != e_y2_prev)

                id_match = olig_x[i, 0] == init_olig_x[j, 0]
                matching = ss or se or es or ee

                if matching and id_match:
                    matchings[i, j] = 1
                else:
                    pass
            else:
                pass

    return matchings


def oligs_orientations(olig_x, olig_y, init_olig_x, init_olig_y, matchings):       
    orientations = np.zeros((olig_x.shape[0], olig_x.shape[0]))
    for i in range(olig_x.shape[0]):
        end_index_1 = np.max(np.nonzero(olig_x[i, :]))
        for j in range(init_olig_x.shape[0]):
            if matchings[i, j] != 0:
                end_index_2 = np.max(np.nonzero(init_olig_x[j, :]))        
                
                s_x1 = olig_x[i, 1]              
                s_y1 = olig_y[i, 1]
                e_x1 = olig_x[i, end_index_1]
                e_y1 = olig_y[i, end_index_1]
                s_x2 = init_olig_x[j, 1]
                s_y2 = init_olig_y[j, 1]
                e_x2 = init_olig_x[j, end_index_2]
                e_y2 = init_olig_y[j, end_index_2]

                s_x1_next = olig_x[i, 2]
                s_y1_next = olig_y[i, 2]
                e_x1_prev = olig_x[i, end_index_1 - 1]
                e_y1_prev = olig_y[i, end_index_1 - 1]
                s_x2_next = init_olig_x[j, 2]
                s_y2_next = init_olig_y[j, 2]
                e_x2_prev = init_olig_x[j, end_index_2 - 1]
                e_y2_prev = init_olig_y[j, end_index_2 - 1]

                ss = (s_x1 == s_x2) and (s_y1 == s_y2) and (s_x1_next != s_x2_next or s_y1_next != s_y2_next)
                se = (s_x1 == e_x2) and (s_y1 == e_y2) and (s_x1_next != e_x2_prev or s_y1_next != e_y2_prev)
                es = (e_x1 == s_x2) and (e_y1 == s_y2) and (e_x1_prev != s_x2_next or e_y1_prev != s_y2_next)
                ee = (e_x1 == e_x2) and (e_y1 == e_y2) and (e_x1_prev != e_x2_prev or e_y1_prev != e_y2_prev)

                if es or ss:
                    orientations[i, j] = 1
                elif ee or se:
                    orientations[i, j] = -1
            else:
                pass

    return orientations


def orient_seed_oligs(matchings, orientations, olig_x, olig_y):
    for i in range(olig_x.shape[0]): # orient all seeds
        end_index = np.max(np.nonzero(olig_x[i, :]))
        if np.sum(matchings[i, :]) == 1 and np.sum(orientations[i, :]) == -1:
             olig_x[i, 1 : np.max(np.nonzero(olig_x[i, :])) + 1] = np.flip(olig_x[i, 1 : np.max(np.nonzero(olig_x[i, :])) + 1])
             olig_y[i, 1 : np.max(np.nonzero(olig_y[i, :])) + 1] = np.flip(olig_y[i, 1 : np.max(np.nonzero(olig_y[i, :])) + 1])        
        else:
            pass

    return olig_x, olig_y


def merge_oligs(matchings, orientations, olig_x, olig_y, init_olig_x, init_olig_y):
    change = 1
    while True: # merge all matching oligomers
        if change == 0:
            break
        else:
            change = 0
            for i in range(olig_x.shape[0]):
                if np.sum(matchings[i, :]) == 1:
                    for j in range(init_olig_x.shape[0]):
                        if i != j:
                            if matchings[i, j] == 1:
                                change = 1
                                last_x_i = np.max(np.nonzero(olig_x[i, :]))
                                last_y_i = np.max(np.nonzero(olig_y[i, :]))
                                last_x_j = np.max(np.nonzero(init_olig_x[j, :]))
                                last_y_j = np.max(np.nonzero(init_olig_y[j, :]))
                                if orientations[i, j] == 1:
                                    olig_x[i, last_x_i : last_x_i + last_x_j] = init_olig_x[j, 1 : last_x_j + 1]
                                    olig_y[i, last_y_i : last_y_i + last_y_j] = init_olig_y[j, 1 : last_y_j + 1]
                                else:
                                    print(np.flip(olig_x[j, 1 : last_x_j + 1]))
                                    olig_x[i, last_x_i : last_x_i + last_x_j] = np.flip(init_olig_x[j, 1 : last_x_j + 1])
                                    olig_y[i, last_y_i : last_y_i + last_y_j] = np.flip(init_olig_y[j, 1 : last_y_j + 1])

                                matchings = oligs_to_match(olig_x, olig_y, init_olig_x, init_olig_y)
                                orientations = oligs_orientations(olig_x, olig_y, init_olig_x, init_olig_y, matchings)
                            else:
                                pass     
                        else:
                            pass
                else:
                    pass
           

    return olig_x, olig_y


curves_x, curves_y = open_tracked_oligo('curve_x.txt', 'curve_y.txt')

matchings = oligs_to_match(curves_x, curves_y, curves_x, curves_y)
orientations = oligs_orientations(curves_x, curves_y, curves_x, curves_y, matchings)
olig_x, olig_y = orient_seed_oligs(matchings, orientations, curves_x, curves_y)

init_matchings = copy.deepcopy(matchings)
init_olig_x, init_olig_y = copy.deepcopy(olig_x), copy.deepcopy(olig_y) 

matchings = oligs_to_match(curves_x, curves_y, init_olig_x, init_olig_y)
orientations = oligs_orientations(curves_x, curves_y, init_olig_x, init_olig_y, matchings)
mt_curve_x, mt_curve_y = merge_oligs(matchings, orientations, olig_x, olig_y, init_olig_x, init_olig_y)
mt_curve_x, mt_curve_y = remove_unnecessary(mt_curve_x, mt_curve_y, init_matchings)

np.savetxt("mt_curve_x.txt", mt_curve_x, '%d', encoding = 'utf-8-sig')
np.savetxt("mt_curve_y.txt", mt_curve_y, '%d', encoding = 'utf-8-sig')

