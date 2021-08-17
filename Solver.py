import math
import numpy as np
import random
import itertools as it
import pprint as pp
from setuptools.command.install_egg_info import install_egg_info


class Solver:
    # crowdsourcing comparisons made by one sky algorithm
    ###### sky_comparisons = set()
    # set_comparisons__point_1__point_2__dim = set()
    # first sorted dim
    # first_sorted_dimension = []
    # #comparisons needed to sort the first dimension
    # set_comparisons__first_dim = set()

    def __init__(self):
        #
        self.set_comparisons__point_1__point_2__dim = set()
        #
        self.num_crowd_phases = 0
        #
        self.first_sorted_dimension = []
        self.set_comparisons__first_dim = set()
        self.num_crowd_phases__first_dim = 0
        #
        #
        return

    #
    ##
    ###
    def get_num_comparisons_performed_by_the_last_method(self):
        return len(self.set_comparisons__point_1__point_2__dim)

    #
    ##
    ###
    def round_robin_tournament_scores(self, list__point, d, worker):
        """
        RR-Turnament
        :param list__point:
        :param d:
        :return:
        """
        scores = 0
        map__point__num_wins = dict(zip(list__point, [0 for x in range(len(list__point))]))
        for i in range(len(list__point)):
            for j in range(i + 1, len(list__point)):
                if worker.is_point_1_smaller_than_point_2(list__point[i], list__point[j], d,
                                                          self.set_comparisons__point_1__point_2__dim):
                    map__point__num_wins[list__point[j]] += 1
                else:
                    map__point__num_wins[list__point[i]] += 1
        return map__point__num_wins

    #
    def get_theoretical_num_comps_in_a_single_round_in_WC_scenario(self, n):
        #
        theoretical_best_value_for_s_in_WC_scenario = 64 if n <= 64 else math.floor(math.sqrt(2 * n))
        #
        theoretical_num_comps_in_this_round_in_WC_scenario = theoretical_best_value_for_s_in_WC_scenario * (
                theoretical_best_value_for_s_in_WC_scenario - 1) / 2
        theoretical_num_comps_in_this_round_in_WC_scenario += (n - theoretical_best_value_for_s_in_WC_scenario)
        #
        return theoretical_num_comps_in_this_round_in_WC_scenario

    #
    ##
    def twoSort(self, list__point, worker, d, map__num_points__best_s=None, try_random_quicksort=False,
                total_amount_of_allowed_comparisons=float("+inf"), tare_of_num_comparisons=0):
        """
        The 2Sort Algorithm by Ajtai at al.
        :param points:
        :param d:
        :param map__num_points__best_s:
        :return: Sorted list of points in DESCENDING order.
        """
        #

        s = math.floor(math.sqrt(2 * len(list__point)))
        if map__num_points__best_s is not None and len(list__point) in map__num_points__best_s:
            s = map__num_points__best_s[len(list__point)]
        #
        #
        if try_random_quicksort and total_amount_of_allowed_comparisons > 0:
            s = 1
        #
        #
        c_num_crowd_phases = 0
        #
        #
        if (len(list__point) > s):
            #
            #
            predicted_num_comparisons_current_round = ((s * (s - 1)) / 2) + (len(list__point) - s)
            #
            ##############################################################
            num_comparisons_performed_so_far = len(
                self.set_comparisons__point_1__point_2__dim) - tare_of_num_comparisons
            #
            if try_random_quicksort and total_amount_of_allowed_comparisons < num_comparisons_performed_so_far + predicted_num_comparisons_current_round:
                if (map__num_points__best_s is not None) and (len(list__point) in map__num_points__best_s):
                    s = map__num_points__best_s[len(list__point)]
                else:
                    s = math.floor(math.sqrt(2 * len(list__point)))
            ##############################################################
            #
            num_comps_t0 = len(self.set_comparisons__point_1__point_2__dim)
            #
            random_subset_of_points = np.random.choice(a=list__point, size=s, replace=False)
            sorted_list_points_desc_num_wins = [x[0] for x in
                                                sorted(self.round_robin_tournament_scores(random_subset_of_points, d,
                                                                                          worker).items(),
                                                       key=lambda kv: -kv[1])]
            pivot = sorted_list_points_desc_num_wins[round(s / 2.)]
            #
            num_comps_t1 = len(self.set_comparisons__point_1__point_2__dim)
            if num_comps_t1 > num_comps_t0: c_num_crowd_phases += 1
            #
            #
            num_comps_t0 = len(self.set_comparisons__point_1__point_2__dim)
            #
            (list__point_1, list__point_2) = self.pivot_splitting(list__point, worker, pivot, d)
            #
            num_comps_t1 = len(self.set_comparisons__point_1__point_2__dim)
            if num_comps_t1 > num_comps_t0: c_num_crowd_phases += 1
            #
            #
            list__point_1, num_crowd_phases_1 = self.twoSort(list__point_1, worker,
                                                             d,
                                                             map__num_points__best_s,
                                                             try_random_quicksort,
                                                             total_amount_of_allowed_comparisons=total_amount_of_allowed_comparisons,
                                                             tare_of_num_comparisons=tare_of_num_comparisons)
            list__point_2, num_crowd_phases_2 = self.twoSort(list__point_2, worker,
                                                             d,
                                                             map__num_points__best_s,
                                                             try_random_quicksort,
                                                             total_amount_of_allowed_comparisons=total_amount_of_allowed_comparisons,
                                                             tare_of_num_comparisons=tare_of_num_comparisons)
            #
            c_num_crowd_phases += max(num_crowd_phases_1, num_crowd_phases_2)
            #
            #
            return list__point_2 + [pivot] + list__point_1, c_num_crowd_phases
        #
        #
        #
        #
        num_comps_t0 = len(self.set_comparisons__point_1__point_2__dim)
        #
        final_result = [x[0] for x in
                        sorted(self.round_robin_tournament_scores(list__point, d, worker).items(),
                               key=lambda kv: (-kv[1], kv[0].at(d)))]
        #
        num_comps_t1 = len(self.set_comparisons__point_1__point_2__dim)
        if num_comps_t1 > num_comps_t0: c_num_crowd_phases += 1
        #
        #
        return final_result, c_num_crowd_phases

    #
    def pivot_splitting(self, list__point, worker, pivot, d):
        """
        return two lists S1 and S2 such that S1 <_d pivot and S2 >_d pivot
        :param points:
        :param pivot:
        :param d:
        :return:
        """
        list__point_1 = []
        list__point_2 = []
        for point in list__point:
            #
            if point == pivot:
                continue
            #
            if worker.is_point_1_smaller_than_point_2(pivot, point, d, self.set_comparisons__point_1__point_2__dim):
                list__point_1.append(point)
            else:
                list__point_2.append(point)
        return (list__point_2, list__point_1)

    #
    ##
    def twoMaxFind(self, list__point, worker, d,
                   map__num_points__best_s=None,
                   try_random_max=False,
                   total_amount_of_allowed_comparisons=float("+inf"), tare_of_num_comparisons=0):
        """

        :param list__point:
        :param worker:
        :param d:
        :param map__num_points__best_s:
        :param try_random_max:
        :param total_amount_of_allowed_comparisons:
        :param tare_of_num_comparisons:
        :return:
        """
        #
        set__max = set()
        #
        #
        initial_s = math.ceil(math.sqrt(len(list__point)))
        s = initial_s
        if map__num_points__best_s is not None and len(list__point) in map__num_points__best_s:
            s = map__num_points__best_s[len(list__point)]
        #
        #
        if try_random_max and total_amount_of_allowed_comparisons > 0:
            s = 1
        #
        #
        c_num_crowd_phases = 0
        #
        #
        set_candidates_as_list_to_max__point = list(list__point)
        while len(set_candidates_as_list_to_max__point) > s:
            #
            #
            predicted_num_comparisons_current_round = ((s * (s - 1)) / 2) + (
                    len(set_candidates_as_list_to_max__point) - s)
            #
            ##############################################################
            num_comparisons_performed_so_far = len(
                self.set_comparisons__point_1__point_2__dim) - tare_of_num_comparisons
            #
            if try_random_max and total_amount_of_allowed_comparisons < num_comparisons_performed_so_far + predicted_num_comparisons_current_round:
                if (map__num_points__best_s is not None) and (
                        len(set_candidates_as_list_to_max__point) in map__num_points__best_s):
                    s = map__num_points__best_s[len(set_candidates_as_list_to_max__point)]
                else:
                    s = initial_s
            ##############################################################
            #
            num_comps_t0 = len(self.set_comparisons__point_1__point_2__dim)
            #
            # extract a random subset of candidates to be the max of size 's'
            random_subset_candidates_as_list_to_max__point = np.random.choice(a=set_candidates_as_list_to_max__point,
                                                                              size=s, replace=False)
            #
            # perform RR on the extracted random subset of candidates to be the max of size 's'
            map__point__RR_num_wins = self.round_robin_tournament_scores(random_subset_candidates_as_list_to_max__point,
                                                                         d, worker)
            #
            num_comps_t1 = len(self.set_comparisons__point_1__point_2__dim)
            if num_comps_t1 > num_comps_t0: c_num_crowd_phases += 1
            #
            #
            # selct the point with maximum number of wins in the extracted random subset of candidates to be the max of size 's'
            local_max_point = max([(point, num_wins) for point, num_wins in map__point__RR_num_wins.items()],
                                  key=lambda x: x[1])[0]
            #
            #
            num_comps_t0 = len(self.set_comparisons__point_1__point_2__dim)
            #
            # compare the local max to all candidates and eliminate all elements that lose to him.
            set_candidates_as_list_to_max__point = self.update_set_of_candidates_for_the_max(local_max_point,
                                                                                             set_candidates_as_list_to_max__point,
                                                                                             d, worker)
            num_comps_t1 = len(self.set_comparisons__point_1__point_2__dim)
            if num_comps_t1 > num_comps_t0: c_num_crowd_phases += 1
            #
            #
            #
        #
        #
        num_comps_t0 = len(self.set_comparisons__point_1__point_2__dim)
        #
        # perform RR on the final set of candidates to be the max of size <='s'
        map__point__RR_num_wins = self.round_robin_tournament_scores(set_candidates_as_list_to_max__point,
                                                                     d, worker)
        num_comps_t1 = len(self.set_comparisons__point_1__point_2__dim)
        if num_comps_t1 > num_comps_t0: c_num_crowd_phases += 1
        #
        # selct the point with maximum number of wins in the extracted random subset of candidates to be the max of size 's'
        num_wins_of_max_point = max([(point, num_wins) for point, num_wins in map__point__RR_num_wins.items()],
                                    key=lambda x: x[1])[1]
        set__max = set(
            [point for point, num_wins in map__point__RR_num_wins.items() if num_wins == num_wins_of_max_point])
        #
        return set__max, c_num_crowd_phases

    #
    def update_set_of_candidates_for_the_max(self, point, set_candidates_as_list_to_max__point, d, worker):
        #
        new_set_candidates_as_list_to_max__point = [point]
        #
        for index in range(len(set_candidates_as_list_to_max__point)):
            #
            c_point = set_candidates_as_list_to_max__point[index]
            #
            if point == c_point:
                continue
            #
            if not (
                    worker.is_point_1_smaller_than_point_2(c_point, point, d,
                                                           self.set_comparisons__point_1__point_2__dim)):
                new_set_candidates_as_list_to_max__point.append(c_point)
        #
        return new_set_candidates_as_list_to_max__point

    #
    ##
    ### OLD
    def all_play_all(self, list__point, worker, use_complete_comparisons=False, reset_set_comparisons=True):
        """
        APA algorithm for skln computation.
        :param list__point:
        :param use_complete_comparisons:
        :return:
        """
        #
        if reset_set_comparisons:
            self.set_comparisons__point_1__point_2__dim = set()
        #
        set_of_dominated_indexes = set()
        self.num_crowd_phases = 1
        #
        for i in range(len(list__point)):
            #
            point_i = list__point[i]
            #
            for j in range(len(list__point)):
                if j == i: continue
                #
                point_j = list__point[j]
                #
                #
                if worker.does_point_1_dominate_point_2(point_j, point_i,
                                                        self.set_comparisons__point_1__point_2__dim,
                                                        perform_all_comparisons=use_complete_comparisons):
                    set_of_dominated_indexes.add(i)

        #
        skln = []
        for i in range(len(list__point)):
            if i not in set_of_dominated_indexes:
                skln.append(list__point[i])
        #
        return skln

    #
    ##
    ### OLD
    def all_play_all_2(self, list__point, worker, use_complete_comparisons=False, reset_set_comparisons=True):
        """
        APA algorithm for skln computation.
        :param list__point:
        :param use_complete_comparisons:
        :return:
        """
        #
        if reset_set_comparisons:
            self.set_comparisons__point_1__point_2__dim = set()
        #
        set_of_dominated_indexes = set()
        self.num_crowd_phases = 1
        #
        list_of_i_indexes = list(range(len(list__point)))
        list_of_j_indexes = list(list_of_i_indexes)
        #
        random.shuffle(list_of_i_indexes)
        for i in list_of_i_indexes:
            #
            point_i = list__point[i]
            #
            #
            random.shuffle(list_of_j_indexes)
            for j in list_of_j_indexes:
                if j == i: continue
                #
                point_j = list__point[j]
                #
                #
                if worker.does_point_1_dominate_point_2(point_j, point_i,
                                                        self.set_comparisons__point_1__point_2__dim,
                                                        perform_all_comparisons=use_complete_comparisons):
                    set_of_dominated_indexes.add(i)
                    break  # BrEaK!!!
        #
        #
        skln = []
        for i in range(len(list__point)):
            if i not in set_of_dominated_indexes:
                skln.append(list__point[i])
        #
        return skln

        #

    ##
    ###
    def repeated_twoMaxFind_for_skln(self, list__point, worker, reset_set_comparisons=True):
        """
        Computes skln by repeating the 2MaxFind algorithm.
        :param list__point:
        :param worker:
        :param map__num_points__best_s:
        :return:
        """
        skln = []
        #
        skln_as_set = set()
        #
        if reset_set_comparisons:
            self.set_comparisons__point_1__point_2__dim = set()
        #
        # mapping point_id to the number of dimensions
        # the point has been selected as maximum ;)
        map__point_id__num_dimensions_in_which_it_has_been_selected_as_max = {}
        temp_list__point = None
        set__dimensions_out_of_the_skln = set()
        num_dimensions = list__point[0].n_dimensions
        map__dimension__inner_list__point = {c_dim: list(list__point) for c_dim in range(num_dimensions)}
        # while len(set__dimensions_out_of_the_skln) < num_dimensions:
        must_continue = True
        while must_continue:
            for d in range(num_dimensions):
                #
                # if d in set__dimensions_out_of_the_skln:
                #    continue

                #
                # 2MF
                set__max, c_num_crowd_phases = self.twoMaxFind(map__dimension__inner_list__point[d], worker, d)
                #
                # print("----")
                # print(" repeated_twoMaxFind_for_skln ", " dim ", d, " |set__max|", len(set__max))
                # for pppp in set__max:
                #    print(" repeated_twoMaxFind_for_skln ", " dim ", pppp)
                # pp.pprint(map__point_id__num_dimensions_in_which_it_has_been_selected_as_max)
                #

                # there_is_at_least_one_max_out_of_skln = False
                for c_max in set__max:
                    #
                    # check if c_max is dominated by at least one point in the current skln
                    c_max_out_of_the_skln = worker.does_exist_a_point_in_collection_that_dominates_the_input_point(
                        skln,
                        c_max, self.set_comparisons__point_1__point_2__dim)
                    #

                    #
                    # is the c_max part of the skln?
                    if not c_max_out_of_the_skln:
                        #
                        # add c_max to the current skln
                        if c_max not in skln_as_set:
                            skln.append(c_max)
                            skln_as_set.add(c_max)
                        # else:
                        #    must_continue = False
                        #
                    #
                    if c_max.ID not in map__point_id__num_dimensions_in_which_it_has_been_selected_as_max:
                        map__point_id__num_dimensions_in_which_it_has_been_selected_as_max[c_max.ID] = 0
                    map__point_id__num_dimensions_in_which_it_has_been_selected_as_max[c_max.ID] += 1
                    if num_dimensions == map__point_id__num_dimensions_in_which_it_has_been_selected_as_max[
                        c_max.ID]:
                        must_continue = False
                        # pp.pprint(map__point_id__num_dimensions_in_which_it_has_been_selected_as_max)

                # remove all c_maxS from the input collection of points
                temp_list__point = [point for point in map__dimension__inner_list__point[d] if point not in set__max]
                map__dimension__inner_list__point[d] = temp_list__point
                #
        #
        #
        return skln

    #
    ##
    def compute_skln_using_all_sorted_components(self, sorted_components):
        skln = []
        #
        dimensions = len(sorted_components)
        while len(sorted_components[0]) > 0:
            new_skln_element = sorted_components[0][0]
            skln.append(new_skln_element)
            set_dominated__point = set(sorted_components[0])
            for d in range(1, dimensions):
                index = sorted_components[d].index(new_skln_element)
                set_dominated__point = set_dominated__point.intersection(set(sorted_components[d][index:]))
            for d in range(0, dimensions):
                sorted_components[d] = [x for x in sorted_components[d] if x not in set_dominated__point]
            #
        return skln

    #
    ##
    ###
    def compute_deterministic_skln(self, list__point):
        skln = []
        #
        sorted_components = []
        #
        # Sort each component independtly.
        dimensions = list__point[0].n_dimensions
        for d in range(dimensions):
            #
            sorted_dimension = sorted(list__point, key=lambda x: (x.at(d)), reverse=True)
            sorted_components.append(sorted_dimension)
            #
        #
        # Compute skln using all sorted components.
        skln = self.compute_skln_using_all_sorted_components(sorted_components)
        #
        return skln

    #
    ##
    ###
    def compute_deterministic_sklnS_of_all_ORDERS(self, list__point):
        map__order__collection_of_points = {}
        map__point_id__order = {}
        list__point_id__order = []
        #
        inner_set__point = set(list__point)
        #
        c_order = 0
        while len(inner_set__point) > 0:
            #
            c_order += 1
            inner_list__point = list(inner_set__point)
            #
            c_skln = self.compute_deterministic_skln(inner_list__point)
            #
            map__order__collection_of_points[c_order] = c_skln
            #
            for c_point in c_skln:
                inner_set__point.remove(c_point)
        #
        for c_order, c_collection_of_points in map__order__collection_of_points.items():
            for c_point in c_collection_of_points:
                map__point_id__order[c_point.ID] = c_order
                list__point_id__order.append((c_point.ID, c_order))
        #
        list__point_id__order.sort(key=lambda x: (x[1], x[0]))
        return list__point_id__order, map__order__collection_of_points, map__point_id__order

    #
    ##
    ###
    def OLD__random_skln(self, list__point, worker, reset_set_comparisons=True):
        """
        No guarantees on ErRor!
        :param list__point:
        :param worker:
        :param reset_set_comparisons:
        :return:
        """
        skln = []
        #
        if reset_set_comparisons:
            self.set_comparisons__point_1__point_2__dim = set()
        #
        list_candidates__point = list__point.copy()
        while len(list_candidates__point) > 0:
            #
            s = random.choice(list_candidates__point)
            #
            s_eliminated = False
            list_candidates__point.remove(s)
            #
            deleted_points = []
            for c_candidate in list_candidates__point:
                if worker.does_point_1_dominate_point_2(s, c_candidate, self.set_comparisons__point_1__point_2__dim):
                    deleted_points.append(c_candidate)
                elif worker.does_point_1_dominate_point_2(c_candidate, s, self.set_comparisons__point_1__point_2__dim):
                    s_eliminated = True
            #
            for deleted_point in deleted_points:
                list_candidates__point.remove(deleted_point)
            if not (s_eliminated):
                skln.append(s)
        #
        return skln

    #
    ##
    ###
    ####
    def divede_et_impera_skln(self, list__point, worker, num_partitions=10, sub_method="2sort_skln",
                              super_method="2sort_skln", map__parameter__value={}, reset_set_comparisons=True):
        skln = []
        #
        if reset_set_comparisons:
            self.set_comparisons__point_1__point_2__dim = set()
        #
        random.shuffle(list__point)
        #
        sigle_partition_size_float = len(list__point) / num_partitions
        sigle_partition_size = math.floor(len(list__point) / num_partitions)
        if sigle_partition_size_float > sigle_partition_size:
            sigle_partition_size += 1
        #
        second_order_list__point = []
        for index in range(0, len(list__point), sigle_partition_size):
            #
            c_chunk_list__point = list__point[index:index + sigle_partition_size]
            #
            c_chunk_skln = []
            if sub_method == "_random_skln":
                c_chunk_skln = self.OLD__random_skln(c_chunk_list__point, worker, reset_set_comparisons=False)
            if sub_method == "2sort_skln":
                #
                map__num_points__best_s = None
                if "sub_method__map__num_points__best_s" in map__parameter__value:
                    map__num_points__best_s = map__parameter__value["sub_method__map__num_points__best_s"]
                #
                c_chunk_skln = self.twoSort_skln(c_chunk_list__point, worker,
                                                 map__num_points__best_s=map__num_points__best_s,
                                                 reset_set_comparisons=False)
            if sub_method == "2lex_skln":
                #
                map__num_points__best_s = None
                if "sub_method__map__num_points__best_s" in map__parameter__value:
                    map__num_points__best_s = map__parameter__value["sub_method__map__num_points__best_s"]
                #
                c_chunk_skln = self.lexicographic_skln(c_chunk_list__point, worker,
                                                       inital_set_comparisons__point_1__point_2__dim=set(),
                                                       map__num_points__best_s=map__num_points__best_s,
                                                       reset_first_sorted_dimension=True, reset_set_comparisons=False)
            #
            second_order_list__point.extend(c_chunk_skln)
            #
            # print()
            # print("  |c_chunk_list__point|", str(len(c_chunk_list__point)))
            # print("  |c_chunk_skln|", str(len(c_chunk_skln)))
            # print("  |self.set_comparisons__point_1__point_2__dim|=",
            #      len(self.set_comparisons__point_1__point_2__dim))
            # print()
            #
        #
        #
        # print()
        # print("|second_order_list__point|", str(len(second_order_list__point)))
        # if len(second_order_list__point) != len(set(second_order_list__point)):
        #    print()
        #    print("len(second_order_list__point) != len(set(second_order_list__point)) !!!!!!")
        #    print("len(second_order_list__point)", len(second_order_list__point))
        #    print("len(set(second_order_list__point))", len(set(second_order_list__point)))
        #    exit(-1)
        #    print()
        # print()
        if super_method == "_random_skln":
            skln = self.OLD__random_skln(second_order_list__point, worker, reset_set_comparisons=False)
        if super_method == "2sort_skln":
            #
            map__num_points__best_s = None
            if "super_method__map__num_points__best_s" in map__parameter__value:
                map__num_points__best_s = map__parameter__value["super_method__map__num_points__best_s"]
            #
            skln = self.twoSort_skln(second_order_list__point, worker, map__num_points__best_s=map__num_points__best_s,
                                     reset_set_comparisons=False)
        if super_method == "2lex_skln":
            #
            map__num_points__best_s = None
            if "super_method__map__num_points__best_s" in map__parameter__value:
                map__num_points__best_s = map__parameter__value["super_method__map__num_points__best_s"]
            #
            skln = self.lexicographic_skln(second_order_list__point, worker,
                                           inital_set_comparisons__point_1__point_2__dim=set(),
                                           map__num_points__best_s=map__num_points__best_s,
                                           reset_first_sorted_dimension=True, reset_set_comparisons=False)
        #
        return skln

    #####################################################################################
    #####################################################################################
    #####################################################################################
    #####################################################################################
    #####################################################################################

    #
    ##
    ###
    def all_play_all_method_for_skln(self, list__point, worker, use_complete_comparisons=False,
                                     reset_set_comparisons=True):
        """
        Compare each element to any other to isolate the ones
        that are not dominated.
        Any ErRoR guarantee??? No.
        :param list__point:
        :param worker:
        :param use_complete_comparisons:
        :param reset_set_comparisons:
        :return:
        """
        #
        if reset_set_comparisons:
            self.set_comparisons__point_1__point_2__dim = set()
        #
        set_of_dominated_indexes = set()
        self.num_crowd_phases = 0
        #
        list_of_indexes = list(range(len(list__point)))
        #
        random.shuffle(list_of_indexes)
        for i, j in it.combinations(list_of_indexes, 2):
            #
            point_i = list__point[i]
            point_j = list__point[j]
            #
            if worker.does_point_1_dominate_point_2(point_j, point_i,
                                                    self.set_comparisons__point_1__point_2__dim,
                                                    perform_all_comparisons=use_complete_comparisons):
                set_of_dominated_indexes.add(i)
            elif worker.does_point_1_dominate_point_2(point_i, point_j,
                                                      self.set_comparisons__point_1__point_2__dim,
                                                      perform_all_comparisons=use_complete_comparisons):
                set_of_dominated_indexes.add(j)
            #
        #
        #
        skln = []
        for i in range(len(list__point)):
            if i not in set_of_dominated_indexes:
                skln.append(list__point[i])
        #
        self.num_crowd_phases = 1
        #
        return skln

    #
    ##
    ###
    def naive_method_for_skln(self, list__point, worker, use_complete_comparisons=False,
                              reset_set_comparisons=True):
        """
        Scan the set of points to isolate the ones that are dominated
        by other points.
        Any ErRoR guarantee??? No.
        :param list__point:
        :param worker:
        :param use_complete_comparisons:
        :param reset_set_comparisons:
        :return:
        """
        #
        if reset_set_comparisons:
            self.set_comparisons__point_1__point_2__dim = set()
        #
        set_of_dominated_indexes = set()
        self.num_crowd_phases = 0
        #
        list_of_i_indexes = list(range(len(list__point)))
        list_of_j_indexes = list(list_of_i_indexes)
        #
        random.shuffle(list_of_i_indexes)
        for i in list_of_i_indexes:
            #
            if i in set_of_dominated_indexes:
                continue
            #
            point_i = list__point[i]
            #
            #
            random.shuffle(list_of_j_indexes)
            num_comps_t0 = len(self.set_comparisons__point_1__point_2__dim)
            for j in list_of_j_indexes:
                if j == i: continue
                #
                if j in set_of_dominated_indexes:
                    continue
                #
                #
                point_j = list__point[j]
                #
                #
                if worker.does_point_1_dominate_point_2(point_j, point_i,
                                                        self.set_comparisons__point_1__point_2__dim,
                                                        perform_all_comparisons=use_complete_comparisons):
                    set_of_dominated_indexes.add(i)
                elif worker.does_point_1_dominate_point_2(point_i, point_j,
                                                          self.set_comparisons__point_1__point_2__dim,
                                                          perform_all_comparisons=use_complete_comparisons):
                    set_of_dominated_indexes.add(j)
            #
            num_comps_t1 = len(self.set_comparisons__point_1__point_2__dim)
            if num_comps_t1 > num_comps_t0: self.num_crowd_phases += 1
            #
        #
        #
        skln = []
        for i in range(len(list__point)):
            if i not in set_of_dominated_indexes:
                skln.append(list__point[i])
        #
        return skln

    #
    ##
    ###
    def lexicographic_skln(self, list__point, worker,
                           inital_set_comparisons__point_1__point_2__dim=set(),
                           map__num_points__best_s=None,
                           reset_first_sorted_dimension=False, reset_set_comparisons=True,
                           try_random_quicksort=False):
        """
        Sove skln problem under the treshold-error model using
        a lexicographic-sort like method.
        :param list__point:
        :param worker:
        :param inital_set_comparisons__point_1__point_2__dim:
        :param map__num_points__best_s:
        :param reset_first_sorted_dimension:
        :return:
        """
        #
        skln = []
        #
        self.num_crowd_phases = 0
        #
        if reset_set_comparisons:
            self.set_comparisons__point_1__point_2__dim = inital_set_comparisons__point_1__point_2__dim
        #
        list_candidates__point = list__point.copy()
        #
        if reset_first_sorted_dimension:
            self.first_sorted_dimension = []
            self.num_crowd_phases__first_dim = 0
        #
        #
        if (len(self.first_sorted_dimension) > 0):
            #
            # print("self.num_crowd_phases__first_dim", self.num_crowd_phases__first_dim)
            self.num_crowd_phases = self.num_crowd_phases__first_dim
            #
            if (len(self.set_comparisons__point_1__point_2__dim) == 0):
                list_candidates__point = self.first_sorted_dimension.copy()
                self.set_comparisons__point_1__point_2__dim = self.set_comparisons__first_dim.copy()
            else:
                # We already have the points sorted according to the first dimension.
                # We already have some performed comparison.
                # We must initialize the list of candidates according to
                # the order in the first dimension.
                set_of_input_points = set(list__point)
                list_candidates__point = []
                for point in self.first_sorted_dimension:
                    if point in set_of_input_points:
                        list_candidates__point.append(point)
                #
        else:
            #
            total_amount_of_allowed_comparisons = float("+inf")
            if try_random_quicksort:
                total_amount_of_allowed_comparisons = int(4 * len(list_candidates__point) ** 1.5) + 1
            #
            tare_of_num_comparisons = len(self.set_comparisons__point_1__point_2__dim)
            list_candidates__point, num_crowd_phases, = self.twoSort(
                list_candidates__point, worker, 0,
                map__num_points__best_s, try_random_quicksort,
                total_amount_of_allowed_comparisons=total_amount_of_allowed_comparisons,
                tare_of_num_comparisons=0)
            #
            num_comparisons_performed_by_2Sort = len(
                self.set_comparisons__point_1__point_2__dim) - tare_of_num_comparisons
            # print("zxcvbnm ", "num_comparisons_performed_by_2Sort=", num_comparisons_performed_by_2Sort, "4n**1.5=", int(4 * len(list_candidates__point) ** 1.5))
            if (num_comparisons_performed_by_2Sort >= int(4 * len(list_candidates__point) ** 1.5)):
                print("asdfghjkl",
                      "(num_comparisons_performed_by_2Sort >= int(4 * len(list_candidates__point) ** 1.5))")

            self.num_crowd_phases__first_dim = num_crowd_phases
            self.num_crowd_phases = num_crowd_phases
        #
        # print("lexicographic_skln", "dim", 0, self.num_crowd_phases)
        #
        while (len(list_candidates__point) > 0):
            #
            num_comps_t0 = len(self.set_comparisons__point_1__point_2__dim)
            #
            m = list_candidates__point[0]
            skln.append(m)
            deleted_pp = [m]
            for i in range(1, len(list_candidates__point)):
                if worker.does_point_1_dominate_point_2(m, list_candidates__point[i],
                                                        self.set_comparisons__point_1__point_2__dim):
                    deleted_pp.append(list_candidates__point[i])
            for p in deleted_pp:
                list_candidates__point.remove(p)
            #
            num_comps_t1 = len(self.set_comparisons__point_1__point_2__dim)
            if num_comps_t1 > num_comps_t0: self.num_crowd_phases += 1
            # print("lexicographic_skln", "phase_2", self.num_crowd_phases)
            #
            #
        return skln

    #
    ##
    ###
    def twoSort_skln(self, list__point, worker, map__num_points__best_s=None, reset_set_comparisons=True,
                     try_random_quicksort=False):
        """
        Compute the skln performing 'd' distinct 2Sort algorithms.
        :param list__point:
        :param map__num_points__best_s:
        :return: skln
        """
        #
        if reset_set_comparisons:
            self.set_comparisons__point_1__point_2__dim = set()
        #
        self.set_comparisons__first_dim = set()
        skln = []
        sorted_components = []
        #
        self.num_crowd_phases = 0
        max__num_crowd_phases = 0
        #
        #
        total_amount_of_allowed_comparisons = float("+inf")
        if try_random_quicksort:
            total_amount_of_allowed_comparisons = int(4 * len(list__point) ** 1.5) + 1
        #
        # apply 2Sort to each component independtly.
        dimensions = list__point[0].n_dimensions
        for d in range(dimensions):
            #
            tare_of_num_comparisons = len(self.set_comparisons__point_1__point_2__dim)
            sorted_dimension, c_dim__num_crowd_phases = self.twoSort(list__point,
                                                                     worker, d,
                                                                     map__num_points__best_s,
                                                                     try_random_quicksort,
                                                                     total_amount_of_allowed_comparisons=total_amount_of_allowed_comparisons,
                                                                     tare_of_num_comparisons=tare_of_num_comparisons)
            max__num_crowd_phases = c_dim__num_crowd_phases if c_dim__num_crowd_phases > max__num_crowd_phases else max__num_crowd_phases
            sorted_components.append(sorted_dimension)
            if d == 0:
                self.set_comparisons__first_dim = self.set_comparisons__point_1__point_2__dim.copy()
                self.num_crowd_phases__first_dim = c_dim__num_crowd_phases
            #
            #
            num_comparisons_performed_by_2Sort = len(
                self.set_comparisons__point_1__point_2__dim) - tare_of_num_comparisons
            # print("zxcvbnm ", "num_comparisons_performed_by_2Sort=", num_comparisons_performed_by_2Sort, "4n**1.5=", int(4 * len(list__point) ** 1.5))
            if (num_comparisons_performed_by_2Sort >= int(4 * len(list__point) ** 1.5)):
                print("asdfghjkl",
                      "(num_comparisons_performed_by_2Sort >= int(4 * len(list_candidates__point) ** 1.5))")
            #
            # print("twoSort_skln", "dim", d, self.num_crowd_phases)
            # print("twoSort_skln", "self.num_crowd_phases__first_dim", self.num_crowd_phases__first_dim)
            #
        self.first_sorted_dimension = sorted_components[0].copy()
        self.num_crowd_phases = max__num_crowd_phases
        #
        # Compute skln using all sorted components.
        skln = self.compute_skln_using_all_sorted_components(sorted_components)
        #
        return skln
