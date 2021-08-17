import itertools
from Point import Point
import math
import pprint as pp


########################################################################################################
########################################################################################################

def does_point_a_completely_dominate_point_b(point_a, point_b, with_equality=True):
    for d in range(len(point_a)):
        if with_equality:
            if point_a[d] <= point_b[d]:
                return False
        else:
            if point_a[d] < point_b[d]:
                return False
    return True


def does_exist_a_point_in_the_set_that_completely_dominate_point_b(set_points, point_b):
    for point_a in set_points:
        point_a_completely_dominate_point_b = does_point_a_completely_dominate_point_b(point_a, point_b)
        if point_a_completely_dominate_point_b:
            return True
    return False


def does_exist_a_point_in_the_set_that_is_completely_dominated_point_b(set_points, point_b):
    for point_a in set_points:
        point_a_completely_dominate_point_b = does_point_a_completely_dominate_point_b(point_b, point_a,
                                                                                       with_equality=False)
        if point_a_completely_dominate_point_b:
            return True
    return False


def compute_belt_around_two_points(s_a, s_b, set__GT_SKLN_points_as_tuples, num_dimensions):
    set__belt = set()
    #
    l = [False, True]
    ll = [l] * num_dimensions
    for choice in itertools.product(*ll):
        c_belt_point = [0.] * num_dimensions
        for d in range(num_dimensions):
            c_belt_point[d] = s_a.at(d) if choice[d] else s_b.at(d)
        #
        if tuple(c_belt_point) in set__GT_SKLN_points_as_tuples:
            continue
        set__belt.add(tuple(c_belt_point))
        #
    #
    return set__belt


def compute_SKLN_belt(GT_SKLN_points):
    #
    belt_points = []
    #
    num_dims = 0
    for s in GT_SKLN_points:
        num_dims = s.n_dimensions
        break
    #
    #
    list__GT_SKLN_points = list(GT_SKLN_points)
    origin_point = Point(components=[0.] * num_dims, ID=str("0"))
    map__dim__sulist_projected_points = {}
    for d in range(num_dims):
        #
        list__GT_SKLN_points.sort(key=lambda p: p.at(d), reverse=True)
        #
        map__dim__sulist_projected_points[d] = [origin_point] + list(list__GT_SKLN_points) + [origin_point]
    #
    #
    set__GT_SKLN_points_as_tuples = set()
    for s in GT_SKLN_points:
        set__GT_SKLN_points_as_tuples.add(tuple(s.vec))
    #
    belt_points = []
    for d in range(num_dims):
        #
        for index in range(len(map__dim__sulist_projected_points[d]) - 1):
            s_a = map__dim__sulist_projected_points[d][index]
            s_b = map__dim__sulist_projected_points[d][index + 1]
            c_set__belt_points_as_tuples = compute_belt_around_two_points(s_a, s_b, set__GT_SKLN_points_as_tuples,
                                                                          num_dims)
            for c_belt_point_as_tuples in c_set__belt_points_as_tuples:
                if does_exist_a_point_in_the_set_that_completely_dominate_point_b(set__GT_SKLN_points_as_tuples,
                                                                                  c_belt_point_as_tuples):
                    continue
                if does_exist_a_point_in_the_set_that_is_completely_dominated_point_b(set__GT_SKLN_points_as_tuples,
                                                                                      c_belt_point_as_tuples):
                    continue
                #
                belt_points.append(c_belt_point_as_tuples)
                #
    #
    set__GT_SKLN_points = set(GT_SKLN_points)
    output_belt_points = []
    c_point_id = 0
    for c_belt_point_components in belt_points:
        c_point_id -= 1
        c_point = Point(components=c_belt_point_components, ID=str(c_point_id))
        output_belt_points.append(c_point)
    #
    return output_belt_points


def compute_SKLN_belt_1(GT_SKLN_points):
    #
    belt_points = []
    #
    num_dims = 0
    for s in GT_SKLN_points:
        num_dims = s.n_dimensions
        break
    #
    map__dim__sulist_projected_values = {d: [0.] for d in range(num_dims)}
    #
    for d in range(num_dims):
        set__projected_values = set()
        for s in GT_SKLN_points:
            set__projected_values.add(s.at(d))
        #
        map__dim__sulist_projected_values[d].extend(set__projected_values)
        map__dim__sulist_projected_values[d].sort()
    #
    #
    set__GT_SKLN_points = set()
    for s in GT_SKLN_points:
        set__GT_SKLN_points.add(tuple(s.vec))
    #
    list_of_lists__dim__sulist_projected_values = []
    for d in range(num_dims):
        list_of_lists__dim__sulist_projected_values.append(map__dim__sulist_projected_values[d])
    for c_point in itertools.product(*list_of_lists__dim__sulist_projected_values):
        #
        if c_point in set__GT_SKLN_points:
            continue
        # print(c_point)
        #
        # check if the current point is COMPLETELY dominating by at least one SKLN point.
        # print("1", does_exist_a_point_in_the_set_that_completely_dominate_point_b(set__GT_SKLN_points, c_point))
        if does_exist_a_point_in_the_set_that_completely_dominate_point_b(set__GT_SKLN_points, c_point):
            continue
        #
        # check if the current point COMPLETELY dominates at least one SKLN point.
        # print("2", does_exist_a_point_in_the_set_that_is_completely_dominated_point_b(set__GT_SKLN_points, c_point))
        if does_exist_a_point_in_the_set_that_is_completely_dominated_point_b(set__GT_SKLN_points, c_point):
            continue
        #
        belt_points.append(c_point)
    #
    #
    set__GT_SKLN_points = set(GT_SKLN_points)
    output_belt_points = []
    c_point_id = 0
    for c_belt_point_components in belt_points:
        c_point_id -= 1
        c_point = Point(components=c_belt_point_components, ID=str(c_point_id))
        output_belt_points.append(c_point)
    #
    return output_belt_points


# from Point import Point
# import pprint as pp
#
# point_id = 1
# p_1 = Point(components=[11., 112., ], ID=str(point_id))
# point_id = 2
# p_2 = Point(components=[21., 22., ], ID=str(point_id))
# point_id = 3
# p_3 = Point(components=[31., 3.,], ID=str(point_id))
# aaaa__GT_SKLN_points = [p_1, p_2, p_3, ]
# belt_points = compute_SKLN_belt(aaaa__GT_SKLN_points)
# pp.pprint(belt_points)
# print("len(belt_points)=", len(belt_points))
# print()
# for s in aaaa__GT_SKLN_points:
#     print(s)
# print("len(aaaa__GT_SKLN_points)=", len(aaaa__GT_SKLN_points))
# print()


#
##
def compute_delta_normalized_l_inf(point_a, point_b, vector_of_absolute_deltas_values):
    relative_to_delta_l_inf_point_a_point_s = 0.
    #
    for d in range(point_a.n_dimensions):
        #
        dist_c_dim = abs(point_a.at(d) - point_b.at(d))
        #
        relative_to_delta_dist_c_dim = dist_c_dim / vector_of_absolute_deltas_values[d]
        #
        relative_to_delta_l_inf_point_a_point_s = relative_to_delta_dist_c_dim if relative_to_delta_dist_c_dim > relative_to_delta_l_inf_point_a_point_s else relative_to_delta_l_inf_point_a_point_s
    #
    return relative_to_delta_l_inf_point_a_point_s


#
##
def compute_delta_normalized_min_distance_over_dimensions(point_a, point_b, vector_of_absolute_deltas_values):
    relative_to_delta_min_distance_over_dimensions_point_a_point_s = float("+inf")
    #
    for d in range(point_a.n_dimensions):
        #
        dist_c_dim = abs(point_a.at(d) - point_b.at(d))
        #
        relative_to_delta_dist_c_dim = dist_c_dim / vector_of_absolute_deltas_values[d]
        #
        relative_to_delta_min_distance_over_dimensions_point_a_point_s = relative_to_delta_dist_c_dim if relative_to_delta_dist_c_dim < relative_to_delta_min_distance_over_dimensions_point_a_point_s else relative_to_delta_min_distance_over_dimensions_point_a_point_s
    #
    return relative_to_delta_min_distance_over_dimensions_point_a_point_s


#
##
###
def compute_SKLN_PRECISION(Algo_SKLN_points, GT_SKLN_points, vector_of_absolute_deltas_values):
    distrib = []
    #
    for point_p in Algo_SKLN_points:
        min_min_distance_over_dimensions = float("+inf")
        for point_s in GT_SKLN_points:
            skip_current_s_point = False
            # for d in range(point_s.n_dimensions):
            # if point_s.at(d) < point_p.at(d):
            #    skip_current_s_point = True
            #    break
            #
            if skip_current_s_point:
                continue
            #
            c_min_distance_over_dimensions_p_s = compute_delta_normalized_min_distance_over_dimensions(point_p, point_s,
                                                                                                       vector_of_absolute_deltas_values)
            #
            min_min_distance_over_dimensions = c_min_distance_over_dimensions_p_s if c_min_distance_over_dimensions_p_s < min_min_distance_over_dimensions else min_min_distance_over_dimensions
            #
        distrib.append(min_min_distance_over_dimensions)
    #
    distrib.sort(reverse=True)
    #
    return distrib


#
##
###
def compute_skln_weighted_recall_distribution(Algo_SKLN_points, GT_SKLN_points,
                                              vector_of_absolute_deltas_values):
    distrib = []
    #
    plus_inf = float("+inf")
    #
    for s in GT_SKLN_points:
        c_min_dist = plus_inf
        for p in Algo_SKLN_points:
            c_relative_to_delta_l_inf = compute_delta_normalized_l_inf(s, p, vector_of_absolute_deltas_values)
            #
            c_min_dist = c_relative_to_delta_l_inf if c_relative_to_delta_l_inf < c_min_dist else c_min_dist
        #
        distrib.append(c_min_dist)
        #
    #
    distrib.sort(reverse=True)
    #
    return distrib


def compute_distribution_of_DIRECTED_minumum_L_inf_distances_from_all_Algo_SKLN_points_to_GT_SKLN_points(
        Algo_SKLN_points, GT_SKLN_points,
        vector_of_absolute_deltas_values, directed_distance=True):
    distrib = []
    #
    plus_inf = float("+inf")
    l_inf_directed_distance_point_a_point_s = 0.
    dist_c_dim = 0.
    #
    skip_current_point_s = False
    for point_a in Algo_SKLN_points:
        min_dist_to_GT_SKLN_points = plus_inf
        #
        c_max_point = [0.] * point_a.n_dimensions
        #
        for point_s in GT_SKLN_points:
            skip_current_point_s = False
            #
            if directed_distance:
                #
                # Check if the current GT_skln point is not smaller than s_point in all dimensions.
                for d in range(point_a.n_dimensions):
                    if point_s.at(d) < point_a.at(d):
                        skip_current_point_s = True
                        break
                if skip_current_point_s:
                    continue
            #
            #
            #
            for d in range(len(c_max_point)):
                c_max_point[d] = point_s.at(d) if point_s.at(d) > c_max_point[d] else c_max_point[d]
            #
            l_inf_directed_distance_point_a_point_s = 0.
            for d in range(point_a.n_dimensions):
                #
                dist_c_dim = point_s.at(d) - point_a.at(d)
                if not directed_distance:
                    dist_c_dim = abs(dist_c_dim)
                #
                relative_to_delta_directed_dist_c_dim = dist_c_dim / vector_of_absolute_deltas_values[d]
                l_inf_directed_distance_point_a_point_s = relative_to_delta_directed_dist_c_dim if relative_to_delta_directed_dist_c_dim > l_inf_directed_distance_point_a_point_s else l_inf_directed_distance_point_a_point_s
                #
                if l_inf_directed_distance_point_a_point_s > min_dist_to_GT_SKLN_points:
                    skip_current_point_s = True
                    break
            #
            if skip_current_point_s:
                continue
            #
            #
            #
            min_dist_to_GT_SKLN_points = l_inf_directed_distance_point_a_point_s if l_inf_directed_distance_point_a_point_s < min_dist_to_GT_SKLN_points else min_dist_to_GT_SKLN_points
        #
        # l_inf distance from edges ;)
        dist_from_edges = plus_inf
        for d in range(point_a.n_dimensions):
            dist_c_dim = c_max_point[d] - point_a.at(d)
            relative_to_delta_dist_c_dim = dist_c_dim / vector_of_absolute_deltas_values[d]
            dist_from_edges = relative_to_delta_dist_c_dim if relative_to_delta_dist_c_dim < dist_from_edges else dist_from_edges
        #
        min_dist_to_GT_SKLN_points = dist_from_edges if dist_from_edges < min_dist_to_GT_SKLN_points else min_dist_to_GT_SKLN_points
        #
        distrib.append(min_dist_to_GT_SKLN_points)
    #
    distrib.sort(reverse=True)
    #
    return distrib


#
##
###
def compute_distribution_distances_from_SKLN_GT_FRONTIER(Algo_SKLN_points, GT_SKLN_points,
                                                         vector_of_absolute_deltas_values):
    #
    distrib = []
    #
    plus_inf = float("+inf")
    #
    for point_a in Algo_SKLN_points:
        #
        distance_from_skln_point_a = 0.
        # min_max_dist_among_dimensions = plus_inf
        #
        for point_s in GT_SKLN_points:
            skip_current_point_s = False
            #
            for d in range(point_a.n_dimensions):
                if point_s.at(d) < point_a.at(d):
                    skip_current_point_s = True
                    break
            if skip_current_point_s:
                continue
            #
            #
            min_dist_among_dimensions = plus_inf
            # max_dist_among_dimensions = -1
            for d in range(point_s.n_dimensions):
                dist_c_dim = point_s.at(d) - point_a.at(d)
                relative_to_delta_dist_c_dim = dist_c_dim / vector_of_absolute_deltas_values[d]
                min_dist_among_dimensions = relative_to_delta_dist_c_dim if relative_to_delta_dist_c_dim < min_dist_among_dimensions else min_dist_among_dimensions
                # max_dist_among_dimensions = relative_to_delta_dist_c_dim if relative_to_delta_dist_c_dim > max_dist_among_dimensions else max_dist_among_dimensions
            #
            #
            distance_from_skln_point_a = min_dist_among_dimensions if min_dist_among_dimensions > distance_from_skln_point_a else distance_from_skln_point_a
            # min_max_dist_among_dimensions = max_dist_among_dimensions if max_dist_among_dimensions < min_distance_from_skln_point_a else min_distance_from_skln_point_a
            #
        #
        distrib.append(distance_from_skln_point_a)
        # distrib.append(min_max_dist_among_dimensions)
    #
    distrib.sort(reverse=True)
    #
    return distrib


#
##
###
def compute_correctness_distribution(Algo_SKLN_points, GT_SKLN_points,
                                     vector_of_absolute_deltas_values):
    #
    distrib = []
    #
    plus_inf = float("+inf")
    #
    for point_t in GT_SKLN_points:
        #
        min__min__distance_from_skln_point_t = plus_inf
        #
        for point_s in Algo_SKLN_points:
            #
            min_dist_over_dimensions = plus_inf
            for d in range(point_s.n_dimensions):
                dist_c_dim = abs(point_t.at(d) - point_s.at(d))
                relative_to_delta_dist_c_dim = dist_c_dim / vector_of_absolute_deltas_values[d]
                min_dist_over_dimensions = relative_to_delta_dist_c_dim if relative_to_delta_dist_c_dim < min_dist_over_dimensions else min_dist_over_dimensions
            #
            #
            min__min__distance_from_skln_point_t = min_dist_over_dimensions if min_dist_over_dimensions < min__min__distance_from_skln_point_t else min__min__distance_from_skln_point_t
            #
        #
        distrib.append(min__min__distance_from_skln_point_t)
    #
    distrib.sort(reverse=True)
    #
    return distrib


##########################################################################
def compute_BOSS_correctness_distribution(Algo_SKLN_points, GT_SKLN_points,
                                          vector_of_absolute_deltas_values):
    #
    distrib = []
    #
    plus_inf = float("+inf")
    #
    for point_t in GT_SKLN_points:
        #
        min_l_inf_distance_from_skln_point_t = plus_inf
        #
        for point_s in Algo_SKLN_points:
            #
            c_l_inf = 0
            for d in range(point_s.n_dimensions):
                dist_c_dim = abs(point_t.at(d) - point_s.at(d))
                relative_to_delta_dist_c_dim = dist_c_dim / vector_of_absolute_deltas_values[d]
                c_l_inf = relative_to_delta_dist_c_dim if relative_to_delta_dist_c_dim > c_l_inf else c_l_inf
            #
            #
            min_l_inf_distance_from_skln_point_t = c_l_inf if c_l_inf < min_l_inf_distance_from_skln_point_t else min_l_inf_distance_from_skln_point_t
            #
        #
        distrib.append(min_l_inf_distance_from_skln_point_t)
    #
    distrib.sort(reverse=True)
    #
    return distrib


def compute_distribution_of_minumum_L_inf_distances_from_all_points_in_collection_A_to_points_in_collection_B(
        collection_of_points_A, collection_of_points_B,
        vector_of_absolute_deltas_values):
    distrib = []
    #
    plus_inf = float("+inf")
    #
    for point_a in collection_of_points_A:
        min_dist_to_points_in_B = plus_inf
        for point_b in collection_of_points_B:
            l_inf_distance_point_a_point_b = 0.
            for d in range(point_a.n_dimensions):
                dist_c_dim = abs(point_a.at(d) - point_b.at(d))
                relative_to_delta_dist_c_dim = dist_c_dim / vector_of_absolute_deltas_values[d]
                l_inf_distance_point_a_point_b = relative_to_delta_dist_c_dim if relative_to_delta_dist_c_dim > l_inf_distance_point_a_point_b else l_inf_distance_point_a_point_b
            #
            min_dist_to_points_in_B = l_inf_distance_point_a_point_b if l_inf_distance_point_a_point_b < min_dist_to_points_in_B else min_dist_to_points_in_B
        distrib.append(min_dist_to_points_in_B)
    #
    distrib.sort(reverse=True)
    #
    return distrib


def compute_distribution_of_SKLN_order(GT__map__point_id__order, collection__point_IDs):
    list__skln_orders = []
    #
    for c_point_id in collection__point_IDs:
        c_order = GT__map__point_id__order.get(c_point_id, -1)
        # c_order = GT__map__point_id__order[c_point_id]
        list__skln_orders.append(c_order)
    #
    list__skln_orders.sort(reverse=True)
    #
    return list__skln_orders


def add_set_metrics(c_record, list__point_IDs, GT_skln_as_set__point_IDs):
    #
    skln_as_set__point_IDs = set(list__point_IDs)
    union_sklnS_as_sorted_list__point_IDs = list(GT_skln_as_set__point_IDs | skln_as_set__point_IDs)
    union_sklnS_as_sorted_list__point_IDs.sort()
    #
    intersection_sklnS_as_sorted_list__point_IDs = list(GT_skln_as_set__point_IDs & skln_as_set__point_IDs)
    intersection_sklnS_as_sorted_list__point_IDs.sort()
    #
    GT_minus_SKLN__as_sorted_list__point_IDs = list(GT_skln_as_set__point_IDs - skln_as_set__point_IDs)
    GT_minus_SKLN__as_sorted_list__point_IDs.sort()
    #
    SKLN_minus_GT__as_sorted_list__point_IDs = list(skln_as_set__point_IDs - GT_skln_as_set__point_IDs)
    SKLN_minus_GT__as_sorted_list__point_IDs.sort()
    #
    c_record.append(union_sklnS_as_sorted_list__point_IDs)
    c_record.append(intersection_sklnS_as_sorted_list__point_IDs)
    c_record.append(GT_minus_SKLN__as_sorted_list__point_IDs)
    c_record.append(SKLN_minus_GT__as_sorted_list__point_IDs)
    return


def update_output_record_exp_crowd_data(c_record_PREFIX, algo_name, c_solver, num_phases, GT_skln_as_set__point_IDs,
                                        GT__map__point_id__order, ALGO_skln_points, GT__list__point_id__order):
    #
    c_record = list(c_record_PREFIX)
    c_record.append(algo_name)
    c_record.append(c_solver.get_num_comparisons_performed_by_the_last_method())
    c_record.append(num_phases)
    c_record.append(len(ALGO_skln_points))
    c_record.append(sorted(list(ALGO_skln_points)))
    #
    add_set_metrics(c_record, ALGO_skln_points, GT_skln_as_set__point_IDs)
    #
    c_record.append(compute_distribution_of_SKLN_order(GT__map__point_id__order, ALGO_skln_points))
    #
    c_record.append(GT__list__point_id__order)
    #
    return c_record


def get_explicit_representation_of_collection_of_points_together_with_a_sorted_list_of_IDS(collection_of_points):
    map__point_id__point_coordinates = {}
    sorted_list__point_IDs = list()
    for point in collection_of_points:
        map__point_id__point_coordinates[point.ID] = tuple(point.vec)
        sorted_list__point_IDs.append(point.ID)
    sorted_list__point_IDs.sort()
    return map__point_id__point_coordinates, sorted_list__point_IDs


#
##
###
def update_output_record(c_record_PREFIX, algo_name, c_solver, num_phases, GT_skln_as_set__point_IDs,
                         GT__map__point_id__order, GT_skln_points, ALGO_skln_points, deltas, GT__list__point_id__order,
                         check_accuracy_and_correcctness=True):
    #
    c_record = list(c_record_PREFIX)
    c_record.append(algo_name)
    c_record.append(c_solver.get_num_comparisons_performed_by_the_last_method())
    c_record.append(num_phases)
    c_record.append(len(ALGO_skln_points))
    map__point_id__point_coordinates, sorted_list__point_IDs = get_explicit_representation_of_collection_of_points_together_with_a_sorted_list_of_IDS(
        ALGO_skln_points)
    c_record.append(map__point_id__point_coordinates)
    c_record.append(sorted_list__point_IDs)
    #
    add_set_metrics(c_record, sorted_list__point_IDs, GT_skln_as_set__point_IDs)
    #
    c_record.append(compute_distribution_of_SKLN_order(GT__map__point_id__order, sorted_list__point_IDs))
    #
    # distribution_of_DIRECTED_minumum_L_inf_distances_from_all_Algo_SKLN_points_to_GT_SKLN_points = compute_distribution_of_DIRECTED_minumum_L_inf_distances_from_all_Algo_SKLN_points_to_GT_SKLN_points(        ALGO_skln_points, GT_skln_points + belt_GT_skln_points, deltas, directed_distance=True)
    # c_record.append(distribution_of_DIRECTED_minumum_L_inf_distances_from_all_Algo_SKLN_points_to_GT_SKLN_points)
    #
    #
    #
    # ACCURACY
    distribution_of_distances_from_all_Algo_SKLN_points_to_GT_SKLN_points = compute_distribution_distances_from_SKLN_GT_FRONTIER(
        ALGO_skln_points, GT_skln_points, deltas)
    #
    if len(distribution_of_distances_from_all_Algo_SKLN_points_to_GT_SKLN_points) == 0:  #### REMOVE!!!!!
        distribution_of_distances_from_all_Algo_SKLN_points_to_GT_SKLN_points = [1000]  #### REMOVE!!!!!
        print("!!!!!!!! len(distribution_of_distances_from_all_Algo_SKLN_points_to_GT_SKLN_points) == 0")
    #
    if distribution_of_distances_from_all_Algo_SKLN_points_to_GT_SKLN_points[0] > 2.:
        print("!!!!!!!! distribution_of_distances_from_all_Algo_SKLN_points_to_GT_SKLN_points[0] := ",
              distribution_of_distances_from_all_Algo_SKLN_points_to_GT_SKLN_points[0])
    #
    # for i in range(len(distribution_of_distances_from_all_Algo_SKLN_points_to_GT_SKLN_points)):  #### REMOVE!!!!!
    #    distribution_of_distances_from_all_Algo_SKLN_points_to_GT_SKLN_points[i] = min(distribution_of_distances_from_all_Algo_SKLN_points_to_GT_SKLN_points[i], 1.99)  #### REMOVE!!!!!

    #
    c_record.append(distribution_of_distances_from_all_Algo_SKLN_points_to_GT_SKLN_points)
    #
    c_record.append(distribution_of_distances_from_all_Algo_SKLN_points_to_GT_SKLN_points[0])
    #
    if check_accuracy_and_correcctness and distribution_of_distances_from_all_Algo_SKLN_points_to_GT_SKLN_points[
        0] >= 2.:
        print("  ACCURACY OUT OF DELTA! ")
        c_record.append("ACCURACY OUT OF DELTA :( " + str(
            distribution_of_distances_from_all_Algo_SKLN_points_to_GT_SKLN_points[0]))
        raise Exception("ACCURACY OUT OF DELTA :( " + str(
            distribution_of_distances_from_all_Algo_SKLN_points_to_GT_SKLN_points[0]))
    #
    # CORRECTNESS
    distribution_of_distances_for_CORRECTNESS = compute_correctness_distribution(
        ALGO_skln_points, GT_skln_points, deltas)
    # print("\n\n")
    # pp.pprint(distribution_of_distances_for_CORRECTNESS)
    # pp.pprint(compute_correctness_distribution(GT_skln_points, GT_skln_points, deltas))
    # print("\n\n")
    #
    #
    if len(distribution_of_distances_for_CORRECTNESS) == 0:  #### REMOVE!!!!!
        distribution_of_distances_for_CORRECTNESS = [1000]  #### REMOVE!!!!!
        print("!!!!!!!! len(distribution_of_distances_for_CORRECTNESS) == 0")
    #
    if distribution_of_distances_for_CORRECTNESS[0] > 2.:
        print("!!!!!!!! distribution_of_distances_for_CORRECTNESS[0] := ",
              distribution_of_distances_for_CORRECTNESS[0])
    #
    # for i in range(len(distribution_of_distances_for_CORRECTNESS)):  #### REMOVE!!!!!
    #    distribution_of_distances_for_CORRECTNESS[i] = min(distribution_of_distances_for_CORRECTNESS[i], 1.99)  #### REMOVE!!!!!
    #
    c_record.append(distribution_of_distances_for_CORRECTNESS)
    #
    c_record.append(distribution_of_distances_for_CORRECTNESS[0])
    #
    if check_accuracy_and_correcctness and distribution_of_distances_for_CORRECTNESS[0] >= 2.:
        print("  CORRECTNESS OUT OF DELTA! ")
        c_record.append("CORRECTNESS OUT OF DELTA :( " + str(
            distribution_of_distances_for_CORRECTNESS[0]))
        raise Exception("CORRECTNESS OUT OF DELTA :( " + str(distribution_of_distances_for_CORRECTNESS[0]))
    #
    #
    #
    c_record.append(str(GT__list__point_id__order))
    #
    return c_record


def compute_precision_and_recall(Algo_skln_as_set__point_IDs, GT_skln_as_set__point_IDs):
    #
    map__metric_name__value = {}
    #
    num_TP = len(Algo_skln_as_set__point_IDs & GT_skln_as_set__point_IDs)
    prec = 0
    if len(Algo_skln_as_set__point_IDs) > 0:
        prec = num_TP / len(Algo_skln_as_set__point_IDs)
    recall = 0
    if len(GT_skln_as_set__point_IDs) > 0:
        recall = num_TP / len(GT_skln_as_set__point_IDs)
    #
    map__metric_name__value["precision"] = prec
    map__metric_name__value["recall"] = recall
    #
    return map__metric_name__value


def compute_Adapted_nDCG_at_GT_skln_size(distribution_of_Algo_skln_points_according_to_GT_skln_orders, GT_skln_size):
    #
    Ideal_nDCG_at_GT_skln_size = GT_skln_size
    #
    distribution_of_Algo_skln_points_according_to_GT_skln_orders.sort()
    #
    DCG_at_GT_skln_size = 0.
    k = min(GT_skln_size, len(distribution_of_Algo_skln_points_according_to_GT_skln_orders))
    for index in range(0, k):
        c_order = distribution_of_Algo_skln_points_according_to_GT_skln_orders[index]
        DCG_at_GT_skln_size += 1. / math.log2(c_order + 1)
    #
    Adapted_nDCG_at_GT_skln_size = DCG_at_GT_skln_size / Ideal_nDCG_at_GT_skln_size
    return Adapted_nDCG_at_GT_skln_size


def compute_Adapted_nDCG_at_Algo_skln_size(distribution_of_Algo_skln_points_according_to_GT_skln_orders,
                                           GT__list_cuples_id__skln_order):
    #
    k = len(distribution_of_Algo_skln_points_according_to_GT_skln_orders)
    #
    Ideal_nDCG_at_GT_skln_size = 0.
    GT__list_cuples_id__skln_order.sort(key=lambda x: (x[1], x[0]))
    for index in range(0, k):
        c_order = GT__list_cuples_id__skln_order[index][1]
        Ideal_nDCG_at_GT_skln_size += 1. / math.log2(c_order + 1)
    #
    if Ideal_nDCG_at_GT_skln_size == 0:
        return 0.
    #
    #
    distribution_of_Algo_skln_points_according_to_GT_skln_orders.sort()
    #
    DCG_at_GT_skln_size = 0.
    for index in range(0, k):
        c_order = distribution_of_Algo_skln_points_according_to_GT_skln_orders[index]
        DCG_at_GT_skln_size += 1. / math.log2(c_order + 1)
    #
    Adapted_nDCG_at_GT_skln_size = DCG_at_GT_skln_size / Ideal_nDCG_at_GT_skln_size
    return Adapted_nDCG_at_GT_skln_size


def compute_Adapted_nDCG_at_k(Algo_skln_as_set__point_IDs, GT__map__point_id__skln_order):
    return

########################################################################################################
########################################################################################################
