import sys
import pprint as pp
import math
import random as rnd
import csv

from Point import Point


##############################################


###
###
###
def fetch_data_form_file_with_crowd_results(name_of_file_with_crowd_results, num_features=3):
    list_of_points = []
    map__point_1__point_2__feature__list_judgments = {}
    #
    # {(p1, p2, d): res}
    map__point_id__point = {}
    #
    input_file = open(name_of_file_with_crowd_results, 'r', encoding="utf-8")
    input_file_csv_reader = csv.reader(input_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_NONE)
    for line in input_file_csv_reader:
        # print(str(line))
        c_feature = int(line[0])
        c_id_1 = int(line[1])
        c_id_2 = int(line[2])
        c_list_of_judgments = [int(single_judgment) for single_judgment in line[3].split(",")]
        #
        if c_id_1 not in map__point_id__point:
            map__point_id__point[c_id_1] = Point(components=[None] * num_features, ID=c_id_1)
        if c_id_2 not in map__point_id__point:
            map__point_id__point[c_id_2] = Point(components=[None] * num_features, ID=c_id_2)
        #
        c_point_1 = map__point_id__point[c_id_1]
        c_point_2 = map__point_id__point[c_id_2]
        map__point_1__point_2__feature__list_judgments[(c_point_1, c_point_2, c_feature)] = c_list_of_judgments
        #
    input_file.close()
    #
    for c_point_id, c_point in map__point_id__point.items():
        list_of_points.append(c_point)
    #
    return list_of_points, map__point_1__point_2__feature__list_judgments


###
###
###
def extract_already_performed_comparisons(map__point_1__point_2__feature__list_judgments):
    # already_performed_comparisons
    # {(p2, p1, d): res}
    # self.comparisons.update({(p1, p2, d): res})
    map__point_1__point_2__feature__judgment = {}
    for point_1__point_2__feature, c_list_of_judgments in map__point_1__point_2__feature__list_judgments.items():
        #
        rnd.shuffle(c_list_of_judgments)
        final_judgment = None
        for c_judgment in c_list_of_judgments:
            #
            # res = p1.at(d) < p2.at(d)
            if c_judgment == -1:
                final_judgment = True  ### CORRECT ONE!!!
                break
            if c_judgment == +1:
                final_judgment = False  ### CORRECT ONE!!!
                break
            #
        if final_judgment is None:
            final_judgment = rnd.choice((True, False))
        #
        map__point_1__point_2__feature__judgment[point_1__point_2__feature] = final_judgment
        #
    return map__point_1__point_2__feature__judgment


def extract_already_performed_comparisons_RANDOMLY(map__point_1__point_2__feature__list_judgments):
    # already_performed_comparisons
    # {(p2, p1, d): res}
    # self.comparisons.update({(p1, p2, d): res})
    map__point_1__point_2__feature__judgment = {}
    for point_1__point_2__feature, c_list_of_judgments in map__point_1__point_2__feature__list_judgments.items():
        #### c_list_of_judgments = map__point_1__point_2__feature__list_judgments[point_1__point_2__feature]
        #
        c_judgment = rnd.choice(c_list_of_judgments)
        if c_judgment == 0:
            c_judgment = rnd.choice([-1, 1])
        #
        # res = p1.at(d) < p2.at(d)
        if c_judgment == -1:
            c_judgment = True  ### CORRECT ONE!!!
        else:
            c_judgment = False  ### CORRECT ONE!!!
        map__point_1__point_2__feature__judgment[point_1__point_2__feature] = c_judgment
    return map__point_1__point_2__feature__judgment


#
##
###
def get_GROUND_TRUTH_for_AMT_dataset():
    #
    map__set_of_features__point__GT_skln_order = {frozenset({0, 1}): {2: 2, 8: 1, 28: 2, 30: 1, 31: 3},
                                                  frozenset({0, 2}): {8: 1, 28: 2},
                                                  frozenset({1, 2}): {2: 1},
                                                  frozenset({0, 1, 2}): {2: 1, 8: 1, 28: 2, 30: 1, 31: 3, 35: 3, 55: 3}}
    #
    map__set_of_features__list_cuples_id_GT_skln_order = {}
    for c_set_of_features in map__set_of_features__point__GT_skln_order:
        #
        default_skln_order_value = 1 + max(skln_order for GT_point_id, skln_order in
                                           map__set_of_features__point__GT_skln_order[c_set_of_features].items())
        #
        map__set_of_features__list_cuples_id_GT_skln_order[c_set_of_features] = []
        for c_id in range(1, 100 + 1):
            c_skln_order = map__set_of_features__point__GT_skln_order[c_set_of_features].get(c_id,
                                                                                             default_skln_order_value)
            map__set_of_features__list_cuples_id_GT_skln_order[c_set_of_features].append((c_id, c_skln_order))

        #
    return map__set_of_features__point__GT_skln_order, map__set_of_features__list_cuples_id_GT_skln_order


#
##
###
def get_GROUND_TRUTH_for_AMT_dataset__OLD():
    #
    map__set_of_features__point__GT_skln_order = {frozenset({0, 1}): {2: 2, 8: 1, 28: 2, 30: 1, 31: 3},
                                                  frozenset({0, 2}): {8: 1, 28: 2},
                                                  frozenset({1, 2}): {2: 1},
                                                  frozenset({0, 1, 2}): {2: 1, 8: 1, 28: 2, 30: 1, 31: 3, 35: 3, 55: 3}}
    #
    map__set_of_features__list_cuples_id_GT_skln_order = {}
    c_set_of_features = frozenset({0, 1, 2})
    map__set_of_features__list_cuples_id_GT_skln_order[c_set_of_features] = []
    for c_id in range(1, 100 + 1):
        c_skln_order = map__set_of_features__point__GT_skln_order[frozenset({0, 1, 2})].get(c_id, 4)
        map__set_of_features__list_cuples_id_GT_skln_order[c_set_of_features].append((c_id, c_skln_order))

    #
    return map__set_of_features__point__GT_skln_order, map__set_of_features__list_cuples_id_GT_skln_order
