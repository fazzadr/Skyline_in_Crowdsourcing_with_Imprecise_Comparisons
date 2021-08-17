import csv
import pprint as pp
import random
import time
from datetime import datetime

import itertools

from Point import Point
from Solver import Solver
from Worker import Worker


def add_keys_to_map__point__feature__set_dominated_points_for_sure(map__point__feature__set_dominated_points_for_sure,
                                                                   feature, id):
    if id not in map__point__feature__set_dominated_points_for_sure:
        map__point__feature__set_dominated_points_for_sure[id] = {}
    if feature not in map__point__feature__set_dominated_points_for_sure[id]:
        map__point__feature__set_dominated_points_for_sure[id][feature] = set()
    return


def update_the_sets_of_dominated_points_for_sure(map__point__feature__set_dominated_points_for_sure, feature, id_1,
                                                 id_2, list__judgments):
    add_keys_to_map__point__feature__set_dominated_points_for_sure(map__point__feature__set_dominated_points_for_sure,
                                                                   feature, id_1)
    add_keys_to_map__point__feature__set_dominated_points_for_sure(map__point__feature__set_dominated_points_for_sure,
                                                                   feature, id_2)
    #
    num_wins_id_1 = 0
    num_wins_id_2 = 0
    num_ties = 0
    for c_judgment in list__judgments:
        if c_judgment == 0:
            num_ties += 1
        elif c_judgment == 1:
            num_wins_id_1 += 1
        elif c_judgment == -1:
            num_wins_id_2 += 1
        else:
            print("ERROR IN PARSING JUDGMENTS!!")
            exit(-1)
    #
    if num_wins_id_1 > num_wins_id_2 + num_ties:
        map__point__feature__set_dominated_points_for_sure[id_1][feature].add(id_2)
    if num_wins_id_2 > num_wins_id_1 + num_ties:
        map__point__feature__set_dominated_points_for_sure[id_2][feature].add(id_1)
    #
    return


#
##
###
def load_AMD_data_for_precise_skln():
    #
    set__feture_id_1__id_2__debug_only = set()
    #
    set__features = set()
    set__points = set()
    map__point__feature__set_dominated_points_for_sure = {}
    #
    input_file_name = "./AMT_dataset/Raw_AMT_results.txt"
    input_file_handler = open(input_file_name, "r", 1000000)
    csv_reader = csv.reader(input_file_handler, delimiter=';', quoting=csv.QUOTE_NONE)
    for record in csv_reader:
        # print(record)
        #
        c_feature = int(record[0])
        c_id_1 = int(record[1])
        c_id_2 = int(record[2])
        c_list__judgments = [int(judgment) for judgment in record[3].split(",")]
        #
        # print(c_feature, c_id_1, c_id_2, c_list__judgments)
        #
        if c_id_1 <= c_id_2:
            set__feture_id_1__id_2__debug_only.add((c_feature, c_id_1, c_id_2))
        else:
            set__feture_id_1__id_2__debug_only.add((c_feature, c_id_2, c_id_1))
        #
        set__features.add(c_feature)
        #
        set__points.add(c_id_1)
        set__points.add(c_id_2)
        #
        update_the_sets_of_dominated_points_for_sure(map__point__feature__set_dominated_points_for_sure, c_feature,
                                                     c_id_1, c_id_2, c_list__judgments)
        #
    input_file_handler.close()
    #
    #

    map__subset_of_features__point__set_of_points_that_dominates_or_could_dominate_him = {}
    for c_subset_of_features_size in range(2, len(set__features) + 1):
        for c_subset_of_features_as_tuple in itertools.combinations(set__features, c_subset_of_features_size):
            #
            c_subset_of_features = frozenset(c_subset_of_features_as_tuple)
            #
            map__point__set_of_points_that_dominates_or_could_dominate_him = {}
            for c_point in set__points:
                #
                c_set_of_dominating_or_potentially_dominating_points = set(set__points)
                c_set_of_dominating_or_potentially_dominating_points.remove(c_point)
                #
                for c_feature in c_subset_of_features:
                    c_feature_c_set_of_dominating_or_potentially_dominating_points = (
                            set__points - map__point__feature__set_dominated_points_for_sure[c_point][c_feature])
                    c_set_of_dominating_or_potentially_dominating_points &= c_feature_c_set_of_dominating_or_potentially_dominating_points
                #
                map__point__set_of_points_that_dominates_or_could_dominate_him[
                    c_point] = c_set_of_dominating_or_potentially_dominating_points
            #
            map__subset_of_features__point__set_of_points_that_dominates_or_could_dominate_him[
                c_subset_of_features] = map__point__set_of_points_that_dominates_or_could_dominate_him
            #
    #
    return set__features, set__points, map__point__feature__set_dominated_points_for_sure, map__subset_of_features__point__set_of_points_that_dominates_or_could_dominate_him, set__feture_id_1__id_2__debug_only


set__features, set__points, map__point__feature__set_dominated_points_for_sure, map__subset_of_features__point__set_of_points_that_dominates_or_could_dominate_him, set__feture_id_1__id_2__debug_only = load_AMD_data_for_precise_skln()

print()
print("set__features")
pp.pprint(set__features)
print()
print("set__points")
pp.pprint(set__points)
print()
# print("map__point__feature__set_dominated_points_for_sure")
# pp.pprint(map__point__feature__set_dominated_points_for_sure)
# print()

# print()
# print("map__point__set_of_points_that_dominates_or_could_dominate_him")
# pp.pprint(map__point__set_of_points_that_dominates_or_could_dominate_him)
# print()


print()
print()
print()

map__set_of_features__point__GT_skln_order = {}

for c_subset_of_features, c_map__point__set_of_points_that_dominates_or_could_dominate_him in map__subset_of_features__point__set_of_points_that_dominates_or_could_dominate_him.items():
    #
    print("======= gnfiuernlb ===============================================")
    print("c_subset_of_features= ", str(c_subset_of_features))
    #
    map__set_of_features__point__GT_skln_order[c_subset_of_features] = {}
    #
    map__point__set_of_points_that_dominates_or_could_dominate_him = dict(
        c_map__point__set_of_points_that_dominates_or_could_dominate_him)
    c_level = 0
    while len(map__point__set_of_points_that_dominates_or_could_dominate_him) > 0:
        c_level += 1
        #
        set__points_not_dominated = set()
        for c_point_1, c_set_of_points_that_dominates_or_could_dominate_him in map__point__set_of_points_that_dominates_or_could_dominate_him.items():
            if len(c_set_of_points_that_dominates_or_could_dominate_him) == 0:
                set__points_not_dominated.add(c_point_1)
        #
        print()
        print("-----------------------------")
        print()
        print("c_level= ", c_level)
        print("c_subset_of_features= ", str(c_subset_of_features))
        print()
        print("set__points_not_dominated")
        pp.pprint(set__points_not_dominated)
        print("|set__points_not_dominated|=", len(set__points_not_dominated))
        print()
        #
        for s_point in set__points_not_dominated:
            map__set_of_features__point__GT_skln_order[c_subset_of_features][s_point] = c_level
        #
        for point in set__points_not_dominated:
            del map__point__set_of_points_that_dominates_or_could_dominate_him[point]
        for c_point_1 in map__point__set_of_points_that_dominates_or_could_dominate_him:
            map__point__set_of_points_that_dominates_or_could_dominate_him[c_point_1] -= set__points_not_dominated
        #
        if len(set__points_not_dominated) == 0:
            break
        #
    print("======================================================")
    print()
    print()
    print()
    print()

print()
print("map__set_of_features__point__GT_skln_order")
pp.pprint(map__set_of_features__point__GT_skln_order)
print()
# # DUMP the dominance graph on file
# output_file_name = "./AMT_dataset__DOMINANCE_GRAPH.csv"
# output_file_handler = open(output_file_name, "w", 1000000)
# csv_writer = csv.writer(output_file_handler, delimiter=',', quoting=csv.QUOTE_NONE)
# csv_writer.writerow(["SOURCE", "TARGET", "interaction", "directed"])
# for c_point_1, c_set_of_points_that_dominates_or_could_dominate_him in map__point__set_of_points_that_dominates_or_could_dominate_him.items():
#     for c_point_2 in c_set_of_points_that_dominates_or_could_dominate_him:
#         csv_writer.writerow([c_point_2, c_point_1, "pp",True])
# output_file_handler.close()


# complete_set__feture_id_1__id_2__debug_only = set()
# for f in set__features:
#     for p_1, p_2 in itertools.combinations(set__points, 2):
#         complete_set__feture_id_1__id_2__debug_only.add((f, p_1, p_2))
#
# print()
# print("complete_set__feture_id_1__id_2__debug_only - set__feture_id_1__id_2__debug_only")
# pp.pprint(complete_set__feture_id_1__id_2__debug_only - set__feture_id_1__id_2__debug_only)
# print("set__feture_id_1__id_2__debug_only - complete_set__feture_id_1__id_2__debug_only")
# pp.pprint(set__feture_id_1__id_2__debug_only - complete_set__feture_id_1__id_2__debug_only)
#
# print()
# print("|set__feture_id_1__id_2__debug_only|         =", len(set__feture_id_1__id_2__debug_only))
# print("|complete_set__feture_id_1__id_2__debug_only|=", len(complete_set__feture_id_1__id_2__debug_only))
