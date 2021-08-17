import math


###############################################

###
def num_comparisons_All_Play_All(num_elements, num_features):
    num_comparisons = int(num_features * ((num_elements * (num_elements - 1)) / 2))
    return num_comparisons


###
def num_comparisons_for_Lower_Bound_with_2_Max_Finding(num_elements, num_features):
    f_1 = 3. * (2. ** (1. / 3.)) / 4.
    f_2 = 3. / 2.
    num_elements_minus_one = float(num_elements - 1)
    LB_num_comparisons = float(num_features) * (
            f_1 * num_elements_minus_one ** (4. / 3.) - f_2 * num_elements_minus_one)
    LB_num_comparisons = int(math.floor(LB_num_comparisons))
    return LB_num_comparisons


###
def min_num_comparisons_for_Output_Sensitive_Lower_Bound(num_elements, num_features):
    min__LB_num_comparisons = (num_features - 1) * num_elements + \
                              (num_features / 2.) * ((num_features / 2.) - 1) + 1. / 4.
    return int(math.floor(min__LB_num_comparisons))


###
def num_comparisons_for_Output_Sensitive_Lower_Bound(num_elements, num_features, size_of_skln):
    LB_num_comparisons = (num_elements - size_of_skln) * (2 * (size_of_skln - 1) + num_features) + (
            size_of_skln * (size_of_skln - 1))
    return LB_num_comparisons


###
def num_comparisons_for_Output_Sensitive_Lower_Bound_with_TRANSITIVITY(num_elements, num_features, size_of_skln):
    LB_num_comparisons = (num_elements - size_of_skln) * num_features + (size_of_skln - 1) * 2
    return LB_num_comparisons


###
def max_between_Lower_Bound_with_2_Max_Finding_and_Output_Sensitive_Lower_Bound(num_elements, num_features,
                                                                                size_of_skln):
    lb_1 = num_comparisons_for_Lower_Bound_with_2_Max_Finding(num_elements, num_features)
    lb_2 = num_comparisons_for_Output_Sensitive_Lower_Bound(num_elements, num_features, size_of_skln)
    max_LB = int(max(lb_1, lb_2))
    return max_LB


###
def min_between_Lower_Bound_with_2_Max_Finding_and_Output_Sensitive_Lower_Bound(num_elements, num_features,
                                                                                size_of_skln):
    lb_1 = num_comparisons_for_Lower_Bound_with_2_Max_Finding(num_elements, num_features)
    lb_2 = num_comparisons_for_Output_Sensitive_Lower_Bound(num_elements, num_features, size_of_skln)
    min_LB = int(min(lb_1, lb_2))
    return min_LB

###############################################
