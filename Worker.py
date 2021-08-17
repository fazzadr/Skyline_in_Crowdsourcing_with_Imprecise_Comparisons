import math
import random


class Worker:
    # crowdsourcing comparisons performed
    # comparisons = {}
    # vector of delta values for each dimension
    # deltas = []
    # string describing the type of oracle
    # worker_policy = 'standard'  # "always_right" # "always_wrong"

    # crowdsourcing comparisons made by one sky algorithm
    # sky_comparisons = set()
    # first sorted dim
    # first_sorted_dim = []
    # # of comparisons needed to sort the first dimension
    # first_dim_sorting_comparisons = set()

    ###
    def __init__(self, deltas, worker_policy, already_performed_comparisons=None):
        #
        self.comparisons = {}
        if already_performed_comparisons != None:
            self.comparisons = already_performed_comparisons
        self.deltas = deltas
        self.worker_policy = worker_policy
        # self.sky_comparisons = set()
        # self.first_sorted_dim = []
        # self.first_dim_sorting_comparisons = set()
        #
        return

    #
    ##
    ##
    def is_point_1_smaller_than_point_2(self, p1, p2, d, set_comparisons__point_1__point_2__dim):
        # def always_right(d, p1, p2):
        #    return p1.at(d) < p2.at(d)
        #
        # def always_wrong(d, p1, p2):
        #    return not (p1.at(d) < p2.at(d))

        # print('oracle type: '+self.o_type)
        if int(p1.ID) < int(p2.ID):
            if (p1, p2, d) not in self.comparisons:
                if (math.fabs(p1.at(d) - p2.at(d))) < self.deltas[d]:
                    if (self.worker_policy == 'always_right'):
                        # res = always_right(d, p1, p2)
                        res = p1.at(d) < p2.at(d)
                    elif (self.worker_policy == 'always_wrong'):
                        # res = always_wrong(d, p1, p2)
                        res = not (p1.at(d) < p2.at(d))
                    else:
                        coin = random.random()
                        if coin > 0.5:
                            res = True
                        else:
                            res = False
                    self.comparisons.update({(p1, p2, d): res})
                    set_comparisons__point_1__point_2__dim.add((p1, p2, d))
                    return res
                else:
                    res = p1.at(d) < p2.at(d)
                    self.comparisons.update({(p1, p2, d): res})
                    set_comparisons__point_1__point_2__dim.add((p1, p2, d))
                    return res
            else:
                set_comparisons__point_1__point_2__dim.add((p1, p2, d))
                return self.comparisons[(p1, p2, d)]

        else:
            if (p2, p1, d) not in self.comparisons:
                if (math.fabs(p1.at(d) - p2.at(d))) < self.deltas[d]:
                    if (self.worker_policy == 'always_right'):
                        # res = always_right(d, p2, p1)
                        res = p2.at(d) < p1.at(d)
                    elif (self.worker_policy == 'always_wrong'):
                        # res = always_wrong(d, p2, p1)
                        res = not (p2.at(d) < p1.at(d))
                    else:
                        coin = random.random()
                        if coin > 0.5:
                            res = True
                        else:
                            res = False
                    self.comparisons.update({(p2, p1, d): res})
                    set_comparisons__point_1__point_2__dim.add((p2, p1, d))
                    return not (res)
                else:
                    res = p2.at(d) < p1.at(d)
                    self.comparisons.update({(p2, p1, d): res})
                    set_comparisons__point_1__point_2__dim.add((p2, p1, d))
                    return not (res)
            else:
                set_comparisons__point_1__point_2__dim.add((p2, p1, d))
                return not (self.comparisons[(p2, p1, d)])

    #
    #
    #
    # deltaOracle(self, point_a, point_b, d)
    #
    # if    dist_d(point_a, point_b) < deltas[d]  ==> arbirary output
    # else                              ==> point_a <_d point_b
    def ___is_point_1_smaller_than_point_2(self, point_1, point_2, d, set_comparisons__point_1__point_2__dim):
        """
        if   dist_d(point_1, point_2) < deltas[d]  ==> return an arbirary boolean answer,
        else                                       ==> return (point_a <_d point_b)
        :param point_1:
        :param point_2:
        :param d:
        :return:
        """
        #
        point_a = point_1
        point_b = point_2
        to_invert = False
        if not (int(point_a.ID) < int(point_b.ID)):
            point_a = point_2
            point_b = point_1
            to_invert = True
        #
        result = None
        if (point_a, point_b, d) not in self.comparisons:
            if (math.fabs(point_a.at(d) - point_b.at(d))) < self.deltas[d]:
                #
                if (self.worker_policy == 'always_right'):
                    result = (point_a.at(d) < point_b.at(d))
                elif (self.worker_policy == 'always_wrong'):
                    result = not (point_a.at(d) < point_b.at(d))
                else:
                    coin = random.random()
                    if coin > 0.5:
                        result = True
                    else:
                        result = False
                #
                self.comparisons.update({(point_a, point_b, d): result})
                set_comparisons__point_1__point_2__dim.add((point_a, point_b, d))
                #
                if to_invert:
                    result = not result
                return result
            else:
                result = point_a.at(d) < point_b.at(d)
                #
                self.comparisons.update({(point_a, point_b, d): result})
                set_comparisons__point_1__point_2__dim.add((point_a, point_b, d))
                #
                if to_invert:
                    result = not result
                #
                return result
        else:
            set_comparisons__point_1__point_2__dim.add((point_a, point_b, d))
            return self.comparisons[(point_a, point_b, d)]
        #
        return result

    #
    ##
    ###
    def does_point_1_dominate_point_2(self, point_1, point_2, set_comparisons__point_1__point_2__dim,
                                      perform_all_comparisons=False):
        """
        Return True if point_1 dominates point_2 according to worker policy
        i.e. if for all d, point_2 <_d point_1.
        :param point_1:
        :param point_2:
        :return:
        """
        point_1_dominate_point_2 = True
        for c_dimension in range(point_1.n_dimensions):
            if self.is_point_1_smaller_than_point_2(point_1, point_2, c_dimension,
                                                    set_comparisons__point_1__point_2__dim):
                point_1_dominate_point_2 = False
                if not perform_all_comparisons:
                    return point_1_dominate_point_2
        return point_1_dominate_point_2

    #
    ##
    ###
    def __does_point_1_dominate_point_2_performing_all_comparisons(self, point_1, point_2,
                                                                   set_comparisons__point_1__point_2__dim):
        """
        Performing a comparison for each dimension,
        the method returns True if point_1 dominates point_2 according to worker policy
        i.e. if for all d, point_2 <_d point_1.
        :param point_1:
        :param point_2:
        :return:
        """
        point_1_dominate_point_2 = True
        #
        for c_dimension in range(point_1.n_dimensions):
            if self.is_point_1_smaller_than_point_2(point_1, point_2, c_dimension,
                                                    set_comparisons__point_1__point_2__dim):
                point_1_dominate_point_2 = False
        #
        return point_1_dominate_point_2

    #
    ##
    ###
    def does_exist_a_point_in_collection_that_dominates_the_input_point(self, col__point, point_1,
                                                                        set_comparisons__point_1__point_2__dim,
                                                                        perform_all_comparisons=False):
        """
        Return True if exists at least one point in the collection
        that dominates point_1 according to the worker policy
        i.e. if exists q in col__point : forall d, p <_d q
        :param col__point:
        :param point_1:
        :return:
        """
        for c_point in col__point:
            if self.does_point_1_dominate_point_2(c_point, point_1, set_comparisons__point_1__point_2__dim,
                                                  perform_all_comparisons=perform_all_comparisons):
                return True
        return False

    #
    ##
    ###
    def less_lex(self, p1, p2):
        """
        DUNNO :\
        :param p1:
        :param p2:
        :return:
        """
        return self.is_point_1_smaller_than_point_2(p1, p2, 0)

    #
    ##
    ###
    def max_lex(self, list__point):
        """
        DUNNO :\
        :param list__point:
        :return:
        """
        maximum = list__point[0]
        for i in range(1, len(list__point)):
            if self.less_lex(maximum, list__point[i]):
                maximum = list__point[i]
        return maximum
