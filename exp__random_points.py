import csv
import pprint as pp
import random
import time
from datetime import datetime
import itertools

from Point import Point
from Solver import Solver
from Worker import Worker
from WorkersDatasetSelector import *
from Bounds import *

from RandomPointsDatasetGenerator import create_set_of_random_points_in_unitary_positive_hypersphere_sector
from RandomPointsDatasetGenerator import create_set_of_grid_points_in_triangle

from Metrics import *


########################################################################################################
########################################################################################################


def create_features_axes(list_of_points):
    all_points = {}
    all_points["feature_1"] = []
    all_points["feature_2"] = []
    for point in list_of_points:
        all_points["feature_1"].append(point.vec[0])
        all_points["feature_2"].append(point.vec[1])
    return all_points


def plot_skln_line(list_of_points, color='k', linestyle=":", alpha=1, plot_errors=False, deltas=None):
    #
    # Compute the True-Real skln of the input set of points ;)
    # c_skln = trueSkyline(list_of_points)
    temp_solver = Solver()
    c_skln = temp_solver.compute_deterministic_skln(list_of_points)
    #
    c_true_skln_points = create_features_axes(c_skln)
    #
    _pf_X_Y = []
    for i in range(len(c_true_skln_points["feature_1"])):
        _pf_X_Y.append([c_true_skln_points["feature_1"][i], c_true_skln_points["feature_2"][i]])
    _pf_X_Y.sort(key=lambda x: (x[1], -x[0]), reverse=True)
    # pp.pprint(_pf_X_Y)
    #
    pf_X_Y = []
    for i in range(len(_pf_X_Y) - 1):
        pf_X_Y.append(_pf_X_Y[i])
        par_x = _pf_X_Y[i][0]
        par_y = _pf_X_Y[i + 1][1]
        pf_X_Y.append([par_x, par_y])
    pf_X_Y.append(_pf_X_Y[-1])
    #
    pf_X = [x for x, y in pf_X_Y]
    pf_Y = [y for x, y in pf_X_Y]
    pf_X = [0] + pf_X
    pf_Y = [pf_Y[0]] + pf_Y
    pf_X.append(pf_X[-1])
    pf_Y.append(0)
    # plt.plot(pf_X, pf_Y, color='k', alpha=0.25)
    plt.plot(pf_X, pf_Y, color=color, linestyle=linestyle, alpha=alpha)
    #
    #
    if not plot_errors:
        return
    #
    #
    pf_X__2_delta_under = [max(x - 2. * deltas[0], 0) for x in pf_X]
    pf_Y__2_delta_under = [max(y - 2. * deltas[1], 0) for y in pf_Y]
    plt.plot(pf_X__2_delta_under, pf_Y__2_delta_under, color='g', alpha=0.15)
    pf_X__2_delta_above = [max(x + 2. * deltas[0], 0) for x in pf_X]
    pf_Y__2_delta_above = [max(y + 2. * deltas[1], 0) for y in pf_Y]
    pf_X__2_delta_above[0] = 0
    pf_Y__2_delta_above[-1] = 0
    # plt.plot(pf_X__2_delta_above, pf_Y__2_delta_above, color='g', alpha=0.1)
    #
    pf_X__1_delta_under = [max(x - 1. * deltas[0], 0) for x in pf_X]
    pf_Y__1_delta_under = [max(y - 1. * deltas[1], 0) for y in pf_Y]
    plt.plot(pf_X__1_delta_under, pf_Y__1_delta_under, color='b', alpha=0.15)
    pf_X__1_delta_above = [max(x + 1. * deltas[0], 0) for x in pf_X]
    pf_Y__1_delta_above = [max(y + 1. * deltas[1], 0) for y in pf_Y]
    pf_X__1_delta_above[0] = 0
    pf_Y__1_delta_above[-1] = 0
    # plt.plot(pf_X__1_delta_above, pf_Y__1_delta_above, color='b', alpha=0.1)
    #
    #
    return


###
###
###
def create_map__num_points__best_s__from_file(input_file_for_creating__map__num_points__best_s, max_n=5000):
    map__num_points__best_s = dict()
    #
    input_file = open(input_file_for_creating__map__num_points__best_s, 'r', encoding="utf-8")
    input_file_csv_reader = csv.reader(input_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_NONE)
    for line in input_file_csv_reader:
        n = int(line[0])
        best_s = int(line[1])
        map__num_points__best_s[n] = best_s
    #
    input_file.close()
    return map__num_points__best_s


########################################################################################################
########################################################################################################

# list_of_features = []

plot_the_points = True
#
num_samples_for_each_input_configuration = 20  ## Because we parallelized with 5 cores ;)
#
# all_worker_policies = ['always_right', 'standard', 'always_wrong']
# worker_policy = 'always_right'
worker_policy = 'standard'
# worker_policy = 'always_wrong'
single_delta = 0.1
#


list__num_points = []
# list__num_points = list__num_points + list(range(10, 100, 10))
# list__num_points = list__num_points + list(range(100, 1000, 100))
# list__num_points = list__num_points + list(range(1000, 5000 + 1, 1000))
#
# list__num_dims = [2, 3, 4]
#
#

#
#
#####################################
### TEST AND DEBUG CONFIGURATIONS ###
num_samples_for_each_input_configuration = 10
list__num_points = [100, 500, 1000]
list__num_dims = [2]
#####################################

#
# list__num_points = list__num_points + list(range(100, 500 + 1, 100))
#
list__input_configurations = [list__num_points, list__num_dims]
# for choice in itertools.product(*ll):
list__num_points__num_dimensions = []
for c_num_points__num_dims in itertools.product(*list__input_configurations):
    list__num_points__num_dimensions.append(c_num_points__num_dims)
# pp.pprint(list__num_points__num_dimensions)


# HCOMP20
plots_output_directory_as_string = "./output/plot__"  ##

#
input_file_for_creating__map__num_points__best_s = "./n__best_s__num_comps_WC__num_comps_WC_2sqrt_n__improvement_ratio___mapping__n__max_number_of_comparisons_WC__10000.csv"
map__num_points__best_s = create_map__num_points__best_s__from_file(input_file_for_creating__map__num_points__best_s)
map__num_points__s_equal_to_3 = {num_points: 3 for num_points in map__num_points__best_s}
map__num_points__s_equal_to_1 = {num_points: 1 for num_points in map__num_points__best_s}
# map__num_points__best_s = None

c_time = str(datetime.now()).replace("-", "_").replace(" ", "__").replace(":", "_").replace(".", "_")
outout_file_name = "./output/" + "output_SKLN_EXP__RANDOM__" + "worker_policy_" + worker_policy + "__" + str(
    c_time) + ".tsv"
output_file_handler = open(outout_file_name, "w", 1000000)
csv_writer_skln = csv.writer(output_file_handler, delimiter='\t', quoting=csv.QUOTE_ALL)
output_header = ["original_workers_file_name",

                 "unique_workers",
                 "consider_partially_suitable_workers",

                 "worker_policy",
                 "single_delta",

                 "list_of_features",
                 "num_dimensions",
                 "num_points",

                 "APA_num_comparisons",
                 "num_comparisons_for_Lower_Bound_with_2_Max_Finding",
                 "min_num_comparisons_for_Output_Sensitive_Lower_Bound",
                 "num_comparisons_for_Output_Sensitive_Lower_Bound_with_TRANSITIVITY",
                 "num_comparisons_for_Output_Sensitive_Lower_Bound",
                 "min_between_Lower_Bound_with_2_Max_Finding_and_Output_Sensitive_Lower_Bound",
                 "max_between_Lower_Bound_with_2_Max_Finding_and_Output_Sensitive_Lower_Bound",

                 "GT_skln_SIZE",
                 "GT_skln_EXPLICIT_POINTS_REPRESENTATION",
                 "GT_skln_SORTED_LIST_POINTS_IDS",

                 "Algo",
                 "Algo_num_comparisons",
                 "Algo_num_phases ",
                 "Algo_skln_SIZE",
                 "Algo_skln_EXPLICIT_POINTS_REPRESENTATION",
                 "Algo_skln_SORTED_LIST_POINTS_IDS",

                 "union_Algo_skln_GT_skln",
                 "intersection_Algo_skln_GT_skln",
                 "points_in_Algo_skln_and_NOT_GT_skln",
                 "points_in_GT_skln_and_NOT_Algo_skln",

                 "sorted_distribution_of_Algo_skln_points_according_to_GT_skln_orders",

                 "sorted_distribution_of_distances_of_Algo_skln_points_from_the_GT_skln_FRONTIER",
                 "MAX_distance_of_Algo_skln_points_from_the_GT_skln_FRONTIER",

                 "sorted_distribution_CORRECTNESS_values",
                 "MAX_CORRECTNESS_value",

                 "list_of_couples__point_id__GT_skln_order",
                 ]
csv_writer_skln.writerow(output_header)
output_file_handler.flush()

original_workers_file_name = "RANDOM_POINTS"
unique_workers = True
consider_partially_suitable_workers = True
output_record_prefix_1 = [original_workers_file_name,
                          unique_workers,
                          consider_partially_suitable_workers,
                          worker_policy,
                          single_delta]

#

#

# list__num_points__num_dimensions = [[30, 4]]
# num_samples_for_each_input_configuration = 1


for _c_sample_number in range(1, num_samples_for_each_input_configuration + 1):
    num_set_of_points_considered_so_far = 0
    for num_points, num_dimensions in list__num_points__num_dimensions:
        print("\n\n")
        #
        #
        c_record_PREFIX = list(output_record_prefix_1)
        c_record_PREFIX.append(str(num_dimensions))
        c_record_PREFIX.append(num_dimensions)
        #
        #
        plot_it = False
        if plot_the_points and num_dimensions == 2:
            plot_it = True
        #
        if plot_it:
            import matplotlib
            import matplotlib.pyplot as plt

        #
        #
        # Creation of the set of multidimensional input points
        # according to the selected characteristics...

        list_of_points = list(
            create_set_of_random_points_in_unitary_positive_hypersphere_sector(num_points, num_dimensions))

        # list_of_points = list(create_set_of_grid_points_in_triangle(num_points))
        # print()
        # print("---------------------")
        # for point in list_of_points: print(point)
        # print("---------------------")
        print("")
        print("len(list_of_points)=", len(list_of_points))
        print("")
        #
        #
        c_record_PREFIX.append(len(list_of_points))
        #
        #
        # #Comparisons-UpperLowerBounds
        c_record_PREFIX.append(num_comparisons_All_Play_All(len(list_of_points), num_dimensions))
        c_record_PREFIX.append(
            num_comparisons_for_Lower_Bound_with_2_Max_Finding(len(list_of_points), num_dimensions))
        c_record_PREFIX.append(
            min_num_comparisons_for_Output_Sensitive_Lower_Bound(len(list_of_points), num_dimensions))
        #
        #
        #

        if len(list_of_points) <= 0:
            continue
        num_set_of_points_considered_so_far += 1

        vector_of_MAXIMUMS = [0.] * num_dimensions
        for p in list_of_points:
            for d in range(p.n_dimensions):
                vector_of_MAXIMUMS[d] = p.at(d) if p.at(d) > vector_of_MAXIMUMS[d] else vector_of_MAXIMUMS[d]
        # deltas = [single_delta] * len(list_of_features)
        deltas = [single_delta * c_MAX for c_MAX in vector_of_MAXIMUMS]
        # if consider_price_as_a_feature:
        #    deltas.append(0.)
        #
        c_worker = Worker(deltas, worker_policy=worker_policy)
        print()
        print("worker_policy", worker_policy)
        print("deltas", deltas)
        print()
        #
        print()
        print("plot_it : " + str(plot_it))
        print("unique_workers : " + str(unique_workers))
        # print("consider_price_as_a_feature : " + str(consider_price_as_a_feature))
        print("consider_partially_suitable_workers : " + str(consider_partially_suitable_workers))
        print("deltas : " + str(deltas))
        print("worker : " + str(c_worker))
        print("num_dimensions : " + str(num_dimensions))
        print()
        print()
        #
        #
        #

        all_points = {}
        if plot_it:
            all_points = create_features_axes(list_of_points)
            plt.scatter(all_points["feature_1"], all_points["feature_2"], facecolors='r', edgecolors='none', alpha=0.1)
            # facecolors='none', edgecolors='c'

        if len(list_of_points) == 0:
            plt.clf()
            continue

        print("ggolf num_dimensions=" + str(num_dimensions))

        #
        c_solver = Solver()
        #
        set__skln = c_solver.compute_deterministic_skln(list_of_points)
        #
        print("len(set__skln)= ", len(set__skln))
        #
        #
        GT__list__point_id__order, GT__map__order__collection_of_points, GT__map__point_id__order = c_solver.compute_deterministic_sklnS_of_all_ORDERS(
            list_of_points)
        GT__map__order__collection_of_points = None
        #
        #
        # #Comparisons Output-Sensitive LowerBounds
        c_record_PREFIX.append(
            num_comparisons_for_Output_Sensitive_Lower_Bound_with_TRANSITIVITY(len(list_of_points), num_dimensions,
                                                                               len(set__skln)))
        c_record_PREFIX.append(
            num_comparisons_for_Output_Sensitive_Lower_Bound(len(list_of_points), num_dimensions, len(set__skln)))
        c_record_PREFIX.append(
            min_between_Lower_Bound_with_2_Max_Finding_and_Output_Sensitive_Lower_Bound(len(list_of_points),
                                                                                        num_dimensions,
                                                                                        len(set__skln)))
        c_record_PREFIX.append(
            max_between_Lower_Bound_with_2_Max_Finding_and_Output_Sensitive_Lower_Bound(len(list_of_points),
                                                                                        num_dimensions,
                                                                                        len(set__skln)))
        #
        c_record_PREFIX.append(len(set__skln))
        map__point_id__point_coordinates, sorted_list__point_IDs = get_explicit_representation_of_collection_of_points_together_with_a_sorted_list_of_IDS(
            set__skln)
        c_record_PREFIX.append(map__point_id__point_coordinates)
        c_record_PREFIX.append(sorted_list__point_IDs)
        #
        GT_skln_as_set__point_IDs = set(sorted_list__point_IDs)

        #

        #
        #
        print()
        # print("set__skln")
        # for p in set__skln: print(p)
        print()
        print("num_features               = ", str(num_dimensions))
        print("num_points_current_instance= ", len(list_of_points))
        print("num_set_of_points_considered_so_far=", num_set_of_points_considered_so_far)
        print("TOTAL_num_set_of_points            =", len(list__num_points__num_dimensions))

        GT_skln_points = list(set__skln)

        # belt_GT_skln_points = compute_SKLN_belt(GT_skln_points)
        belt_GT_skln_points = None

        # true_skln_points = list(set__skln)
        true_skln_points = list(set__skln)
        if plot_it:
            skln = true_skln_points
            # plot_skln_line(list_of_points, color='k', linestyle=":", alpha=1, plot_errors=False, deltas=None)
            plot_skln_line(skln, color='k', linestyle="-", alpha=0.3, plot_errors=True, deltas=deltas)
            #
            true_skln_points = create_features_axes(skln)
            #
            _pf_X_Y = []
            for i in range(len(true_skln_points["feature_1"])):
                _pf_X_Y.append([true_skln_points["feature_1"][i], true_skln_points["feature_2"][i]])
            _pf_X_Y.sort(key=lambda x: (x[1], -x[0]), reverse=True)
            #
            for x, y in _pf_X_Y:
                __x = []
                __y = []
                for _x, _y in [[-deltas[0], -deltas[1]], [-deltas[0], +deltas[1]], [+deltas[0], +deltas[1]],
                               [+deltas[0], -deltas[1]], [-deltas[0], -deltas[1]]]:
                    __x.append(x + _x)
                    __y.append(y + _y)
                # plt.plot(__x, __y, color='b', alpha=0.25)
            for x, y in _pf_X_Y:
                __x = []
                __y = []
                for _x, _y in [[-2. * deltas[0], -2. * deltas[1]], [-2. * deltas[0], +2. * deltas[1]],
                               [+2. * deltas[0], +2. * deltas[1]], [+2. * deltas[0], -2. * deltas[1]],
                               [-2. * deltas[0], -2. * deltas[1]]]:
                    __x.append(x + _x)
                    __y.append(y + _y)
                # plt.plot(__x, __y, color='g', alpha=0.25)
            #
            #
            #
            size = (matplotlib.rcParams['lines.markersize'] ** 2) * 8
            plt.scatter(true_skln_points["feature_1"], true_skln_points["feature_2"], facecolors='none', edgecolors='k',
                        alpha=1, marker="o", s=size, label="true_skln")

            # belt_GT_skln_points___features_axes = create_features_axes(belt_GT_skln_points)
            # plt.scatter(belt_GT_skln_points___features_axes["feature_1"],
            #             belt_GT_skln_points___features_axes["feature_2"],
            #             facecolors='none', edgecolors='k',
            #             alpha=1, marker=".", s=size / 4, label="skln_BELT")

        #
        #
        #

        #
        # Perform all comparisons over all dimensions.
        # skln__apa_complete_comparisons = c_solver.all_play_all(list_of_points, c_worker, use_complete_comparisons=True)

        #
        # 2SortSkln
        skln__2_sort_skln = c_solver.twoSort_skln(list_of_points, c_worker,
                                                  map__num_points__best_s=map__num_points__best_s,
                                                  reset_set_comparisons=True)
        print("len(skln__2_sort_skln)= ", len(skln__2_sort_skln))
        #
        #
        algo_name = "twoSort_skln map__num_points__best_s"
        num_phases = c_solver.num_crowd_phases
        ALGO_skln_points = skln__2_sort_skln
        c_record = update_output_record(c_record_PREFIX, algo_name, c_solver, num_phases, GT_skln_as_set__point_IDs,
                                        GT__map__point_id__order, GT_skln_points, ALGO_skln_points, deltas,
                                        GT__list__point_id__order)
        csv_writer_skln.writerow(c_record)
        output_file_handler.flush()
        #
        #
        if plot_it:
            SD_sky_points = create_features_axes(skln__2_sort_skln)
            plot_skln_line(skln__2_sort_skln, color='b', linestyle="--")
            plt.scatter(SD_sky_points["feature_1"], SD_sky_points["feature_2"], facecolors='k', edgecolors='k', alpha=1,
                        marker="_", label="SD_skln")

        #
        #
        # 2SortSkln NONE
        skln__2_sort_skln = c_solver.twoSort_skln(list_of_points, c_worker,
                                                  map__num_points__best_s=None,
                                                  reset_set_comparisons=True)
        print("len(skln__2_sort_skln NONE)= ", len(skln__2_sort_skln))
        #
        #
        algo_name = "twoSort_skln NONE"
        num_phases = c_solver.num_crowd_phases
        ALGO_skln_points = skln__2_sort_skln
        c_record = update_output_record(c_record_PREFIX, algo_name, c_solver, num_phases, GT_skln_as_set__point_IDs,
                                        GT__map__point_id__order, GT_skln_points, ALGO_skln_points, deltas,
                                        GT__list__point_id__order)
        csv_writer_skln.writerow(c_record)
        output_file_handler.flush()

        #
        # LexSortSkln
        skln__lex_sort_skln = c_solver.lexicographic_skln(list_of_points, c_worker,
                                                          inital_set_comparisons__point_1__point_2__dim=set(),
                                                          map__num_points__best_s=map__num_points__best_s,
                                                          reset_first_sorted_dimension=True, reset_set_comparisons=True)
        print("len(skln__lex_sort_skln)= ", len(skln__lex_sort_skln))
        #
        #
        algo_name = "skln__lex_sort_skln map__num_points__best_s"
        num_phases = c_solver.num_crowd_phases
        ALGO_skln_points = skln__lex_sort_skln
        c_record = update_output_record(c_record_PREFIX, algo_name, c_solver, num_phases, GT_skln_as_set__point_IDs,
                                        GT__map__point_id__order, GT_skln_points, ALGO_skln_points, deltas,
                                        GT__list__point_id__order)
        csv_writer_skln.writerow(c_record)
        output_file_handler.flush()
        #
        #
        if plot_it:
            lex_sky_points = create_features_axes(skln__lex_sort_skln)
            plot_skln_line(skln__lex_sort_skln, color='m', linestyle=":")
            plt.scatter(lex_sky_points["feature_1"], lex_sky_points["feature_2"], facecolors='k', edgecolors='k',
                        alpha=1, marker="|", label="Lex_skln")

        #
        #
        # LexSortSkln NONE
        skln__lex_sort_skln = c_solver.lexicographic_skln(list_of_points, c_worker,
                                                          inital_set_comparisons__point_1__point_2__dim=set(),
                                                          map__num_points__best_s=None,
                                                          reset_first_sorted_dimension=True,
                                                          reset_set_comparisons=True)
        print("len(skln__lex_sort_skln NONE)= ", len(skln__lex_sort_skln))
        #
        #
        algo_name = "skln__lex_sort_skln NONE"
        num_phases = c_solver.num_crowd_phases
        ALGO_skln_points = skln__lex_sort_skln
        c_record = update_output_record(c_record_PREFIX, algo_name, c_solver, num_phases, GT_skln_as_set__point_IDs,
                                        GT__map__point_id__order, GT_skln_points, ALGO_skln_points, deltas,
                                        GT__list__point_id__order)
        csv_writer_skln.writerow(c_record)
        output_file_handler.flush()

        #
        # # Divide et Impra Methods DO NOT WORK!!!!!!!!!!!
        # list__all_divede_et_impera_skln_configurations = [
        #     ("2sort_skln", "2sort_skln"),
        #     ("2sort_skln", "2lex_skln"),
        #     ("2lex_skln", "2sort_skln"),
        #     ("2lex_skln", "2lex_skln"),
        # ]
        # map__parameter__value = {}
        # map__parameter__value["sub_method__map__num_points__best_s"] = map__num_points__best_s
        # map__parameter__value["super_method__map__num_points__best_s"] = map__num_points__best_s
        # for c_sub_method, c_super_method in list__all_divede_et_impera_skln_configurations:
        #     #
        #     # divede_et_impera_skln__twoSort_skln__twoSort_skln
        #     num_partitions = max(int(len(list_of_points) ** 0.5), 3)
        #     c_skln__divede_et_impera_skln = c_solver.divede_et_impera_skln(list_of_points,
        #                                                                    c_worker,
        #                                                                    num_partitions=num_partitions,
        #                                                                    sub_method=c_sub_method,
        #                                                                    super_method=c_super_method,
        #                                                                    map__parameter__value=map__parameter__value,
        #                                                                    reset_set_comparisons=True)
        #     print("len(c_skln__divede_et_impera_skln)= ", len(c_skln__divede_et_impera_skln))
        #     #
        #     #
        #     algo_name = "divede_et_impera_skln__" + c_sub_method + "__" + c_super_method
        #     num_phases = -1
        #     ALGO_skln_points = c_skln__divede_et_impera_skln
        #     c_record = update_output_record(c_record_PREFIX, algo_name, c_solver, num_phases, GT_skln_as_set__point_IDs,
        #                                     GT__map__point_id__order, GT_skln_points, ALGO_skln_points, deltas,
        #                                     belt_GT_skln_points)
        #     csv_writer_skln.writerow(c_record)
        #     output_file_handler.flush()
        #     #
        #

        #

        #

        if plot_it:
            plt.xlabel("feature_1")
            plt.ylabel("feature_2")
            #
            # plt.legend(loc=4)
            # plt.legend()
            # plt.legend(loc='best', frameon=False)
            # plt.legend(loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=2, borderaxespad=0, frameon=True)
            plt.legend(loc='lower left', bbox_to_anchor=(0.0, 1.00), ncol=3, borderaxespad=0, frameon=True)
            # worker_policy
            plot_file_name = str(num_points) + "__" + str(num_dimensions) + "__" + worker_policy + "__" + str(
                single_delta) + "__" + str(random.random()).replace("0.",
                                                                    "") + "__" + str(
                int(round(time.time() * 1000)))
            plt.savefig(plots_output_directory_as_string + plot_file_name + "__.png", dpi=300)
            plt.clf()
        #
        #
        output_file_handler.flush()
        #
output_file_handler.close()
