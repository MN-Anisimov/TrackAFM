import matplotlib.pyplot as plt
import codecs
import numpy as np


def open_tracked_oligo(filename_x, filename_y):
    with codecs.open(filename_x, encoding='utf-8-sig') as f1:
        tracked_oligo_x = np.loadtxt(f1)
    with codecs.open(filename_y, encoding='utf-8-sig') as f2:
        tracked_oligo_y = np.loadtxt(f2)
    
    return tracked_oligo_x, tracked_oligo_y


#def find_correlat(oligo_1_x, oligo_2_x, oligo_1_y, oligo_2_y):
#    length = len(oligo_1)
#    for i in range(length):
#        dist 
#    return


def oligs_to_match(olig_x, olig_y, olig_num, period):
    matchings = np.zeros((olig_num, olig_num))
    orientations = np.zeros((olig_num, olig_num))
    for i in range(olig_num):
        start_index_1 = i * period
        end_index_1 = np.max(np.nonzero(olig_x[start_index_1:start_index_1 + period])) + start_index_1
        for j in range(olig_num):
            if j < i:
                start_index_2 = j * period
                end_index_2 = np.max(np.nonzero(olig_x[start_index_2:start_index_2 + period])) + start_index_2
                
                s_x1 = olig_x[start_index_1]
                s_y1 = olig_y[start_index_1]
                e_x1 = olig_x[end_index_1]
                e_y1 = olig_y[end_index_1]
                s_x2 = olig_x[start_index_2]
                s_y2 = olig_y[start_index_2]
                e_x2 = olig_x[end_index_2]
                e_y2 = olig_y[end_index_2]

                ss = s_x1 == s_x2 and s_y1 == s_y2
                se = s_x1 == e_x2 and s_y1 == e_y2
                es = e_x1 == s_x2 and e_y1 == s_y2
                ee = e_x1 == e_x2 and e_y1 == e_y2
                matching = ss or se or es or ee

                if matching:
                    matchings[i, j] = 1
                    matchings[j, i] = 1
                    if ss:
                        orientations[i, j] = -1
                        orientations[j, i] = -1
                    elif se:
                        orientations[i, j] = -1
                        orientations[j, i] = 1
                    elif es:
                        orientations[i, j] = 1
                        orientations[j, i] = -1
                    else:
                        orientations[i, j] = 1
                        orientations[j, i] = 1
                else:
                    pass
            else:
                pass

    return matchings, orientations


def merge_oligs(matchings, orientations, olig_x, olig_y, olig_num, period):
    merged_x = np.zeros((1, 2000))
    merged_y = np.zeros((1, 2000))
    change = 1
    merged_num = 0
    for i in range(olig_num): # collect all one-sided oligomers
        start_index = i * period
        end_index = np.max(np.nonzero(olig_x[start_index:start_index + period])) + start_index
        if np.sum(matchings[i, :]) == 1 or np.sum(matchings[:, i]) == 1:
            merged_x = np.append(merged_x, np.zeros((1, 2000)), axis = 0)
            merged_y = np.append(merged_y, np.zeros((1, 2000)), axis = 0)
            merged_num = merged_num + 1
            if np.sum(matchings[i, :]) == 1:
                if np.sum(orientations[i, :]) == 1:
                    for j in range(end_index - start_index + 1):
                        merged_x[merged_num, j] = olig_x[start_index + j]
                        merged_y[merged_num, j] = olig_y[start_index + j]
                else:
                    for j in range(end_index - start_index + 1):
                        merged_x[merged_num, j] = olig_x[end_index - j]
                        merged_y[merged_num, j] = olig_y[end_index - j]
            else:
                if np.sum(orientations[:, i]) == 1:
                    for j in range(end_index - start_index + 1):
                        merged_x[merged_num, j] = olig_x[start_index + j]
                        merged_y[merged_num, j] = olig_y[start_index + j]
                else:
                    for j in range(end_index - start_index + 1):
                        merged_x[merged_num, j] = olig_x[end_index - j]
                        merged_y[merged_num, j] = olig_y[end_index - j]
        else:
            pass

    while True: # merge all matching oligomers
        if change == 0:
            break
        else:
            change = 0
            for i in range(merged_x.shape[0] - 1):
                for ii in range(olig_num):
                    start_index = ii * period
                    end_index = np.max(np.nonzero(olig_x[start_index:start_index + period])) + start_index
                    if np.sum(matchings[ii, :]) != 0:
                        is_match_start_x = merged_x[i + 1, np.max(np.nonzero(merged_x[i + 1, :]))] == olig_x[start_index]
                        is_match_end_x = merged_x[i + 1, np.max(np.nonzero(merged_x[i + 1, :]))] == olig_x[end_index]
                        is_match_start_y = merged_y[i + 1, np.max(np.nonzero(merged_y[i + 1, :]))] == olig_y[start_index]
                        is_match_end_y = merged_y[i + 1, np.max(np.nonzero(merged_y[i + 1, :]))] == olig_y[end_index]
                        if (is_match_start_x and is_match_start_y) or (is_match_end_x and is_match_end_y):
                            merge_index_start = np.max(np.nonzero(merged_x[i + 1, :]))
                            for j in range(end_index - start_index + 1):
                                if is_match_start_x and is_match_start_y:
                                   if merged_x[i + 1, merge_index_start - 1] == olig_x[start_index + 1] and merged_y[i + 1, merge_index_start - 1] == olig_y[start_index + 1]:
                                        break
                                   else:
                                        merged_x[i + 1, merge_index_start + j] = olig_x[start_index + j]
                                        merged_y[i + 1, merge_index_start + j] = olig_y[start_index + j]
                                        change = 1
                                else:
                                    if merged_x[i + 1, merge_index_start - 1] == olig_x[end_index - 1] and merged_y[i + 1, merge_index_start - 1] == olig_y[end_index - 1]:
                                        break
                                    else:
                                        merged_x[i + 1, merge_index_start + j] = olig_x[end_index - j]
                                        merged_y[i + 1, merge_index_start + j] = olig_y[end_index - j]
                                        change = 1
                        else:
                            pass
                    else:
                        pass

    return merged_x

#a = numpy.array([1, 2, 3])
#numpy.flip(a)


if __name__ == "__main__":
    (curves_x, curves_y) = open_tracked_oligo('curve_x.txt', 'curve_y.txt')
    period = curves_x.shape[1]
    olig_num = np.max(np.nonzero(curves_x[:, 0])) + 1
    curves_x = curves_x.ravel()
    curves_y = curves_y.ravel()
    (matchings, orientations) = oligs_to_match(curves_x, curves_y, olig_num, period)
    merged_oligomers = merge_oligs(matchings, orientations, curves_x, curves_y, olig_num, period)
    merged_oligomers
