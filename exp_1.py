import csv
import pprint as pp
import random

from Point import Point
from Solver import Solver
from Worker import Worker


########################################################################################################
########################################################################################################

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

#
#
#
#
#
input_file_for_creating__map__num_points__best_s = "/Users/ikki/Dropbox/SKLN_in_CS/sw/2Sort_Analysis/n__best_s__num_comps_WC__num_comps_WC_2sqrt_n__improvement_ratio___mapping__n__max_number_of_comparisons_WC__10000.csv"
map__num_points__best_s = create_map__num_points__best_s__from_file(input_file_for_creating__map__num_points__best_s)
#
number_of_points = 10
list__point = []
list__values_1 = [val for val in range(number_of_points)]
list__values_2 = [val for val in range(number_of_points)]
list__values_3 = [val for val in range(number_of_points)]
list__values_4 = [val for val in range(number_of_points)]
random.shuffle(list__values_1)
random.shuffle(list__values_2)
random.shuffle(list__values_3)
random.shuffle(list__values_4)
# list__point = [Point([list__values_1[index], list__values_2[index]]) for index in range(len(list__values_1))]
list__point = [Point([list__values_1[index],
                      list__values_2[index],
                      list__values_3[index],
                      # list__values_4[index]
                      ]) for index in range(len(list__values_1))]
print()
print("list__point")
# for p in list__point: print(p)
print("|list__point|=", len(list__point))
print()

#
#
d = list__point[0].n_dimensions
deltas = [1.1] * d
worker_policy = "always_right"
# worker_policy = "always_wrong"
# worker_policy = "random"
wrkr = Worker(deltas, worker_policy=worker_policy)
print()
print("worker_policy", worker_policy)
print("deltas", deltas)
print()
list__worker__algo__method = [
    ("compute_deterministic_skln", wrkr, Solver(), "compute_deterministic_skln", None),

    ("apa_complete_comparisons", wrkr, Solver(), "apa_complete_comparisons", None),
    ("apa", wrkr, Solver(), "apa", None),

    ("2sort_skln", wrkr, Solver(), "2sort_skln", {"map__num_points__best_s": None}),
    ("2sort_skln with map__num_points__best_s", wrkr, Solver(), "2sort_skln",
     {"map__num_points__best_s": map__num_points__best_s}),

    ("2lex_skln", wrkr, Solver(), "2lex_skln", {"map__num_points__best_s": None}),
    ("2lex_skln with map__num_points__best_s", wrkr, Solver(), "2lex_skln",
     {"map__num_points__best_s": map__num_points__best_s}),

    # (wrkr, Solver(), "random_skln", None),

    ("divede_et_impera_skln 2sort_skln 2sort_skln", wrkr, Solver(), "divede_et_impera_skln",
     {"sub_method": "2sort_skln", "super_method": "2sort_skln", "sub_method__map__num_points__best_s": None,
      "super_method__map__num_points__best_s": None}),
    ("divede_et_impera_skln 2sort_skln 2lex_skln", wrkr, Solver(), "divede_et_impera_skln",
     {"sub_method": "2sort_skln", "super_method": "2lex_skln", "sub_method__map__num_points__best_s": None,
      "super_method__map__num_points__best_s": None}),
    ("divede_et_impera_skln 2lex_skln 2sort_skln", wrkr, Solver(), "divede_et_impera_skln",
     {"sub_method": "2lex_skln", "super_method": "2sort_skln", "sub_method__map__num_points__best_s": None,
      "super_method__map__num_points__best_s": None}),
    ("divede_et_impera_skln 2lex_skln 2lex_skln", wrkr, Solver(), "divede_et_impera_skln",
     {"sub_method": "2lex_skln", "super_method": "2lex_skln", "sub_method__map__num_points__best_s": None,
      "super_method__map__num_points__best_s": None}),

    (
        "divede_et_impera_skln 2sort_skln 2sort_skln with map__num_points__best_s", wrkr, Solver(),
        "divede_et_impera_skln",
        {"sub_method": "2sort_skln", "super_method": "2sort_skln",
         "sub_method__map__num_points__best_s": map__num_points__best_s,
         "super_method__map__num_points__best_s": map__num_points__best_s}),
    ("divede_et_impera_skln 2sort_skln 2lex_skln with map__num_points__best_s", wrkr, Solver(), "divede_et_impera_skln",
     {"sub_method": "2sort_skln", "super_method": "2lex_skln",
      "sub_method__map__num_points__best_s": map__num_points__best_s,
      "super_method__map__num_points__best_s": map__num_points__best_s}),
    ("divede_et_impera_skln 2lex_skln 2sort_skln with map__num_points__best_s", wrkr, Solver(), "divede_et_impera_skln",
     {"sub_method": "2lex_skln", "super_method": "2sort_skln",
      "sub_method__map__num_points__best_s": map__num_points__best_s,
      "super_method__map__num_points__best_s": map__num_points__best_s}),
    ("divede_et_impera_skln 2lex_skln 2lex_skln with map__num_points__best_s", wrkr, Solver(), "divede_et_impera_skln",
     {"sub_method": "2lex_skln", "super_method": "2lex_skln",
      "sub_method__map__num_points__best_s": map__num_points__best_s,
      "super_method__map__num_points__best_s": map__num_points__best_s}),

]
#


for c_method_description, c_worker, c_solver, c_method, c_method_parameters in list__worker__algo__method:
    print()
    print()
    print("=============================================================")
    print("=============================================================")
    print("=============================================================")
    print(c_worker, c_solver, c_method, c_method_parameters)
    print()

    set__skln = set()

    if c_method == "compute_deterministic_skln":
        set__skln = c_solver.compute_deterministic_skln(list__point)
    if c_method == "apa_complete_comparisons":
        set__skln = c_solver.all_play_all(list__point, c_worker, use_complete_comparisons=True)
        print("|c_worker.comparisons|=", len(c_worker.comparisons))
    if c_method == "apa":
        set__skln = c_solver.all_play_all(list__point, c_worker, use_complete_comparisons=False)

    if c_method == "2sort_skln":
        set__skln = c_solver.twoSort_skln(list__point, c_worker,
                                          map__num_points__best_s=c_method_parameters["map__num_points__best_s"])
    if c_method == "2sort_skln":
        set__skln = c_solver.twoSort_skln(list__point, c_worker,
                                          map__num_points__best_s=c_method_parameters["map__num_points__best_s"])

    if c_method == "2lex_skln":
        set__skln = c_solver.lexicographic_skln(list__point, c_worker,
                                                inital_set_comparisons__point_1__point_2__dim=set(),
                                                map__num_points__best_s=c_method_parameters["map__num_points__best_s"],
                                                reset_first_sorted_dimension=False)
    if c_method == "2lex_skln":
        set__skln = c_solver.lexicographic_skln(list__point, c_worker,
                                                inital_set_comparisons__point_1__point_2__dim=set(),
                                                map__num_points__best_s=c_method_parameters["map__num_points__best_s"],
                                                reset_first_sorted_dimension=False)

    if c_method == "random_skln":
        set__skln = c_solver.OLD__random_skln(list__point, c_worker)

    if c_method == "divede_et_impera_skln":
        num_partitions = max(int(len(list__point) ** 0.5), 1)
        # num_partitions = 2
        set__skln = c_solver.divede_et_impera_skln(list__point, c_worker, num_partitions=num_partitions,
                                                   sub_method=c_method_parameters["sub_method"],
                                                   super_method=c_method_parameters["super_method"])
    #
    #
    #
    print()
    for p in set__skln:
        print(p)
    print("|set__skln|=", len(set__skln))

    print()
    print("c_algo.set_comparisons__point_1__point_2__dim")
    # for p1, p2, dim in alg.set_comparisons__point_1__point_2__dim:
    #    print(p1, p2, dim)
    print("|c_solver.set_comparisons__point_1__point_2__dim|=", len(c_solver.set_comparisons__point_1__point_2__dim),
          c_method, c_method_description)
    print()
    print("c_worker.comparisons")
    # for key, val in worker.comparisons.items():
    #    print(key[0], key[1], key[2], val)
    print("|c_worker.comparisons|=", len(c_worker.comparisons))
    print()
    print("=============================================================")
    print("=============================================================")
    print("=============================================================")
    print()

    print()
    print()
    print()
