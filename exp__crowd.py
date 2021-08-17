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

from CrowdOracle import *

# from RandomPointsDatasetGenerator import create_set_of_random_points_in_unitary_positive_hypersphere_sector
# from RandomPointsDatasetGenerator import create_set_of_grid_points_in_triangle

from Metrics import *


#############################################################################################


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


#


#

#############################################################################################

input_file_for_creating__map__num_points__best_s = "./n__best_s__num_comps_WC__num_comps_WC_2sqrt_n__improvement_ratio___mapping__n__max_number_of_comparisons_WC__10000.csv"
map__num_points__best_s = create_map__num_points__best_s__from_file(input_file_for_creating__map__num_points__best_s)

num_samples = 1000

c_time = str(datetime.now()).replace("-", "_").replace(" ", "__").replace(":", "_").replace(".", "_")
outout_file_name = "./output/" + "output_SKLN_EXP__CROWD__" + str(c_time) + ".tsv"
output_file_handler = open(outout_file_name, "w", 1000000)
csv_writer_skln = csv.writer(output_file_handler, delimiter='\t', quoting=csv.QUOTE_ALL)
output_header = [
    "original_workers_file_name",

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
    "GT_skln_SORTED_LIST_POINTS_IDS",

    "Algo",
    "Algo_num_comparisons",
    "Algo_num_phases",
    "Algo_skln_SIZE",
    "Algo_skln_SORTED_LIST_POINTS_IDS",

    "union_Algo_skln_GT_skln",
    "intersection_Algo_skln_GT_skln",
    "points_in_Algo_skln_and_NOT_GT_skln",
    "points_in_GT_skln_and_NOT_Algo_skln",

    "sorted_distribution_of_Algo_skln_points_according_to_GT_skln_orders",

    "list_of_couples__point_id__GT_skln_order",
]
csv_writer_skln.writerow(output_header)
output_file_handler.flush()

for _ in range(num_samples):
    #
    set_of_all_features = frozenset([0, 1, 2])
    #
    name_of_file_with_crowd_results = "./AMT_dataset/Raw_AMT_results.txt"
    num_dimensions = len(set_of_all_features)
    list_of_points, map__point_1__point_2__feature__list_judgments = fetch_data_form_file_with_crowd_results(
        name_of_file_with_crowd_results, num_features=num_dimensions)
    #
    map__point_1__point_2__feature__judgment = extract_already_performed_comparisons(
        map__point_1__point_2__feature__list_judgments)

    map__set_of_features__point__GT_skln_order, map__set_of_features__list_cuples_id_GT_skln_order = get_GROUND_TRUTH_for_AMT_dataset()
    GT__list__point_id__order = map__set_of_features__list_cuples_id_GT_skln_order[set_of_all_features]

    #
    # print()
    # print(map__set_of_features__point__GT_skln_order)
    # pp.pprint(map__set_of_features__point__GT_skln_order)
    set__skln = set()
    for c_point, c_point_GT_skln_order in map__set_of_features__point__GT_skln_order[set_of_all_features].items():
        if c_point_GT_skln_order == 1:
            set__skln.add(c_point)
    GT_skln_as_set__point_IDs = set(set__skln)
    GT__map__point_id__order = map__set_of_features__point__GT_skln_order[set_of_all_features]

    #
    #
    #

    c_record_PREFIX = [
        name_of_file_with_crowd_results,

        sorted(list(set_of_all_features)),
        num_dimensions,
        len(list_of_points),
    ]

    #
    # #Comparisons-UpperLowerBounds
    c_record_PREFIX.append(num_comparisons_All_Play_All(len(list_of_points), num_dimensions))
    c_record_PREFIX.append(
        num_comparisons_for_Lower_Bound_with_2_Max_Finding(len(list_of_points), num_dimensions))
    c_record_PREFIX.append(
        min_num_comparisons_for_Output_Sensitive_Lower_Bound(len(list_of_points), num_dimensions))
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
    #
    c_record_PREFIX.append(len(set__skln))
    c_record_PREFIX.append(sorted(list(set__skln)))
    #
    #
    #
    #
    #
    #
    # list_of_points = list_of_points[:10]
    #
    deltas = [None] * num_dimensions
    c_worker = Worker(deltas, worker_policy='standard',
                      already_performed_comparisons=map__point_1__point_2__feature__judgment)

    ###############################

    print("")
    print("len(list_of_points)=", len(list_of_points))
    print("")

    print("ggolf num_features=" + str(num_dimensions))

    #
    c_solver = Solver()

    print("num_features               = ", str(num_dimensions))
    print("num_points_current_instance= ", len(list_of_points))

    #
    # 2SortSkln
    skln__2_sort_skln = c_solver.twoSort_skln(list_of_points, c_worker,
                                              map__num_points__best_s=map__num_points__best_s,
                                              reset_set_comparisons=True)
    algo_name = "twoSort_skln map__num_points__best_s"
    num_phases = c_solver.num_crowd_phases
    ALGO_skln_points = sorted([int(p.ID) for p in skln__2_sort_skln])
    print()
    print(algo_name)
    print(algo_name, "skln = ", ALGO_skln_points)
    print(algo_name, "|skln| = ", len(ALGO_skln_points))
    print("num_comparisons=", c_solver.get_num_comparisons_performed_by_the_last_method())
    print("num_phases     =", num_phases)

    c_record = update_output_record_exp_crowd_data(c_record_PREFIX, algo_name, c_solver, num_phases,
                                                   GT_skln_as_set__point_IDs,
                                                   GT__map__point_id__order, ALGO_skln_points,
                                                   GT__list__point_id__order)

    csv_writer_skln.writerow(c_record)
    output_file_handler.flush()
    #
    # 2SortSkln
    skln__2_sort_skln = c_solver.twoSort_skln(list_of_points, c_worker,
                                              map__num_points__best_s=None,
                                              reset_set_comparisons=True)
    algo_name = "twoSort_skln NONE"
    num_phases = c_solver.num_crowd_phases
    ALGO_skln_points = sorted([int(p.ID) for p in skln__2_sort_skln])
    print()
    print(algo_name)
    print(algo_name, "skln = ", ALGO_skln_points)
    print(algo_name, "|skln| = ", len(ALGO_skln_points))
    print("num_comparisons=", c_solver.get_num_comparisons_performed_by_the_last_method())
    print("num_phases     =", num_phases)

    c_record = update_output_record_exp_crowd_data(c_record_PREFIX, algo_name, c_solver, num_phases,
                                                   GT_skln_as_set__point_IDs,
                                                   GT__map__point_id__order, ALGO_skln_points,
                                                   GT__list__point_id__order)

    csv_writer_skln.writerow(c_record)
    output_file_handler.flush()
    #
    # LexSortSkln
    skln__lex_sort_skln = c_solver.lexicographic_skln(list_of_points, c_worker,
                                                      inital_set_comparisons__point_1__point_2__dim=set(),
                                                      map__num_points__best_s=map__num_points__best_s,
                                                      reset_first_sorted_dimension=True, reset_set_comparisons=True)
    algo_name = "skln__lex_sort_skln map__num_points__best_s"
    num_phases = c_solver.num_crowd_phases
    ALGO_skln_points = sorted([int(p.ID) for p in skln__lex_sort_skln])
    print()
    print(algo_name)
    print(algo_name, "skln = ", ALGO_skln_points)
    print(algo_name, "|skln| = ", len(ALGO_skln_points))
    print("num_comparisons=", c_solver.get_num_comparisons_performed_by_the_last_method())
    print("num_phases     =", num_phases)

    c_record = update_output_record_exp_crowd_data(c_record_PREFIX, algo_name, c_solver, num_phases,
                                                   GT_skln_as_set__point_IDs,
                                                   GT__map__point_id__order, ALGO_skln_points,
                                                   GT__list__point_id__order)

    csv_writer_skln.writerow(c_record)
    output_file_handler.flush()
    #
    # LexSortSkln
    skln__lex_sort_skln = c_solver.lexicographic_skln(list_of_points, c_worker,
                                                      inital_set_comparisons__point_1__point_2__dim=set(),
                                                      map__num_points__best_s=None,
                                                      reset_first_sorted_dimension=True, reset_set_comparisons=True)
    algo_name = "skln__lex_sort_skln NONE"
    num_phases = c_solver.num_crowd_phases
    ALGO_skln_points = sorted([int(p.ID) for p in skln__lex_sort_skln])
    print()
    print(algo_name)
    print(algo_name, "skln = ", ALGO_skln_points)
    print(algo_name, "|skln| = ", len(ALGO_skln_points))
    print("num_comparisons=", c_solver.get_num_comparisons_performed_by_the_last_method())
    print("num_phases     =", num_phases)

    c_record = update_output_record_exp_crowd_data(c_record_PREFIX, algo_name, c_solver, num_phases,
                                                   GT_skln_as_set__point_IDs,
                                                   GT__map__point_id__order, ALGO_skln_points,
                                                   GT__list__point_id__order)

    csv_writer_skln.writerow(c_record)
    output_file_handler.flush()
    #
    # ApA
    skln__apa_skln = c_solver.all_play_all(list_of_points, c_worker,
                                           use_complete_comparisons=False,
                                           reset_set_comparisons=True)
    algo_name = "ApA"
    num_phases = c_solver.num_crowd_phases
    ALGO_skln_points = sorted([int(p.ID) for p in skln__apa_skln])
    print()
    print(algo_name)
    print(algo_name, "skln = ", ALGO_skln_points)
    print(algo_name, "|skln| = ", len(ALGO_skln_points))
    print("num_comparisons=", c_solver.get_num_comparisons_performed_by_the_last_method())
    print("num_phases     =", num_phases)

    c_record = update_output_record_exp_crowd_data(c_record_PREFIX, algo_name, c_solver, num_phases,
                                                   GT_skln_as_set__point_IDs,
                                                   GT__map__point_id__order, ALGO_skln_points,
                                                   GT__list__point_id__order)

    csv_writer_skln.writerow(c_record)
    output_file_handler.flush()

    #
    # ApA complete comparisons
    skln__apa_skln = c_solver.all_play_all(list_of_points, c_worker,
                                           use_complete_comparisons=True,
                                           reset_set_comparisons=True)
    algo_name = "ApA complete comparisons"
    num_phases = c_solver.num_crowd_phases
    ALGO_skln_points = sorted([int(p.ID) for p in skln__apa_skln])
    print()
    print(algo_name)
    print(algo_name, "skln = ", ALGO_skln_points)
    print(algo_name, "|skln| = ", len(ALGO_skln_points))
    print("num_comparisons=", c_solver.get_num_comparisons_performed_by_the_last_method())
    print("num_phases     =", num_phases)

    c_record = update_output_record_exp_crowd_data(c_record_PREFIX, algo_name, c_solver, num_phases,
                                                   GT_skln_as_set__point_IDs,
                                                   GT__map__point_id__order, ALGO_skln_points,
                                                   GT__list__point_id__order)

    csv_writer_skln.writerow(c_record)
    output_file_handler.flush()

output_file_handler.flush()
