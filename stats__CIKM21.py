import csv
import pprint as pp
import random
import time
from datetime import datetime
import itertools

from Metrics import *

import numpy as np

from CrowdOracle import *

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sea


##########################################################################################################
##########################################################################################################


def plotter__HCOMP2020(map__algo__distrib,
                       plot_title, y_axis_label, complete_output_file_name):
    #
    # PLOT
    plt.figure(figsize=(6, 6))

    # Create an axes instance
    # ax = plt.subplot(1, 1, 1)
    ax = plt.subplot()

    plt.yticks(fontsize='large')

    lost__plot_titles_with_Ymin0_Ymin1 = ["precision",
                                          "recall",
                                          #
                                          "Adapted_nDCG_at_GT_skln_size",  # "nDCG@|GT_SKLN|",
                                          "Adapted_nDCG_at_Algo_skln_size",  # "nDCG@|Algo_SKLN|",
                                          ]
    #
    if plot_title in lost__plot_titles_with_Ymin0_Ymin1:
        plt.ylim(bottom=0, top=1)

    all_distributions_to_plot = [
        # map__algo__distrib["2SortDims"],
        # map__algo__distrib["2SortDims+"],
        # map__algo__distrib["2LexSort"],
        # map__algo__distrib["2LexSort+"],
        #

        map__algo__distrib["ApA"],
        map__algo__distrib["naive_method_for_skln"],
        #
        #
        map__algo__distrib["twoSort_skln NONE"],
        # map__algo__distrib["twoSort_skln map__num_points__best_s"],
        map__algo__distrib["twoSort_skln map__num_points__best_s RPQS"],
        map__algo__distrib["skln__lex_sort_skln NONE"],
        # map__algo__distrib["skln__lex_sort_skln map__num_points__best_s"],
        map__algo__distrib["skln__lex_sort_skln map__num_points__best_s RPQS"],

    ]

    #
    #
    """
    for index in range(len(all_distributions_to_plot)):
        y = all_distributions_to_plot[index]
        # Add some random "jitter" to the x-axis
        x = np.random.normal(index + 1, 0.04, size=len(y))
        # x = np.random.normal(index + 1, 0.1, size=len(y))
        ax.plot(x, y, 'k.', marker="o", markerfacecolor="k", markeredgecolor="none", alpha=0.5, lw=4.)
    """
    #
    #

    # plt.ylim(bottom=0, top=1)
    # plt.ylim(bottom=0)
    plt.grid(True, linestyle='-',
             color='lightgrey',
             alpha=0.5)

    wwwwwidth = 0.6
    bp = ax.boxplot(all_distributions_to_plot
                    ,
                    widths=(
                        wwwwwidth,
                        wwwwwidth,

                        wwwwwidth,
                        # 0.4,
                        wwwwwidth,

                        wwwwwidth,
                        # 0.4,
                        wwwwwidth,

                    ),
                    showfliers=True)
    #
    # ax.set_xticklabels([
    #     '2SortDims',
    #     '2SortDims+',
    #     '2LexSort',
    #     '2LexSort+',
    # ], rotation=0, fontsize=15, weight='bold')
    ax.set_xticklabels([
        'All-play-All',
        'Naive',

        "SortedDims",  # "'2SortDims',
        # 'SortedDims+',
        'SortedDims++',
        "SingleDim",  # "'2LexSort',
        # 'SingleDim+',
        'SingleDim++',

    ],
        #    rotation=0, fontsize=5.5, weight='bold'
        # rotation=30, fontsize=9, weight='bold'
        rotation=30, fontsize='large', weight='bold'
    )

    #
    #
    set_ylabel_fontsize = 15
    y_label_text = y_axis_label
    print("jkdfghil y_label_text", y_label_text)
    print("jkdfghil y_axis_label", y_axis_label)
    if y_axis_label == "num_comparisons":
        y_label_text = "Cost"
    if y_axis_label == "num_phases":
        y_label_text = "Latency"
    if y_axis_label == "skln_size":
        y_label_text = "Output Skyline Size"
    if y_axis_label == "Adapted_nDCG_at_GT_skln_size":
        y_label_text = "Adapted nDCG at Ground-Truth Skyline Size"
        set_ylabel_fontsize = 14
    if y_axis_label == "Adapted_nDCG_at_Algo_skln_size":
        y_label_text = "Adapted nDCG at Algorithm Skyline Size"
    #
    # ax.set_ylabel(y_label_text, fontsize=15, weight='bold')
    ax.set_ylabel(y_label_text, fontsize=set_ylabel_fontsize, weight='bold')
    # ax.set_xlabel('Algorithm', fontsize=15, weight='bold')
    # ax.set_ylabel(y_label_text, fontsize="x-large", weight='bold')
    # ax.set_xlabel('Algorithm', fontsize=15, weight='bold')

    ## ax.set_title(plot_title, size=15, weight='bold')
    #
    #
    #
    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='black', linewidth=2)

    for cap in bp['caps']:
        cap.set(color='black', linewidth=2)

        ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='red', linewidth=4)

    ## change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        # change outline color
        box.set(color='black', linewidth=3)
    #
    plt.tight_layout()
    plt.savefig(complete_output_file_name)
    #
    return


def plotter__HCOMP2020__whit_LowerBound(map__algo__distrib,
                                        plot_title, y_axis_label, complete_output_file_name):
    #
    # PLOT
    plt.figure(figsize=(6, 6))
    # sea.figure(figsize=(6, 6))

    # Create an axes instance
    # ax = plt.subplot(1, 1, 1)
    ax = plt.subplot()
    # ax = sea.subplot()

    plt.yticks(fontsize='large')

    lost__plot_titles_with_Ymin0_Ymin1 = ["precision",
                                          "recall",
                                          #
                                          "Adapted_nDCG_at_GT_skln_size",  # "nDCG@|GT_SKLN|",
                                          "Adapted_nDCG_at_Algo_skln_size",  # "nDCG@|Algo_SKLN|",
                                          ]
    #
    if plot_title in lost__plot_titles_with_Ymin0_Ymin1:
        plt.ylim(bottom=0, top=1)

    #
    #

    all_distributions_to_plot = [
        # map__algo__distrib["2SortDims"],
        # map__algo__distrib["2SortDims+"],
        # map__algo__distrib["2LexSort"],
        # map__algo__distrib["2LexSort+"],
        #
        #
        # map__algo__distrib["max_between_Lower_Bound_with_2_Max_Finding_and_Output_Sensitive_Lower_Bound"],
        map__algo__distrib["num_comparisons_for_Output_Sensitive_Lower_Bound"],
        #
        ##map__algo__distrib["ApA"],
        map__algo__distrib["naive_method_for_skln"],
        #
        map__algo__distrib["twoSort_skln NONE"],
        # map__algo__distrib["twoSort_skln map__num_points__best_s"],
        map__algo__distrib["twoSort_skln map__num_points__best_s RPQS"],
        map__algo__distrib["skln__lex_sort_skln NONE"],
        # map__algo__distrib["skln__lex_sort_skln map__num_points__best_s"],
        map__algo__distrib["skln__lex_sort_skln map__num_points__best_s RPQS"],

    ]

    #
    #
    """
    for index in range(len(all_distributions_to_plot)):
        y = all_distributions_to_plot[index]
        ### Add some random "jitter" to the x-axis
        x = np.random.normal(index + 1, 0.04, size=len(y))
        ### x = np.random.normal(index + 1, 0.1, size=len(y))
        ax.plot(x, y, 'k.', marker="o", markerfacecolor="k", markeredgecolor="none", alpha=0.5, lw=4.)
    """
    #
    #

    # plt.ylim(bottom=0, top=1)
    # plt.ylim(bottom=0)
    plt.grid(True, linestyle='-',
             color='lightgrey',
             alpha=0.5)
    wwwwwidth = 0.6
    bp = ax.boxplot(all_distributions_to_plot,
                    widths=(
                        wwwwwidth,
                        #
                        # wwwwwidth,
                        wwwwwidth,
                        #
                        wwwwwidth,
                        # 0.4,
                        wwwwwidth,

                        wwwwwidth,
                        # 0.4,
                        wwwwwidth,

                    ),
                    showfliers=True
                    )

    #
    # ax.set_xticklabels([
    #     '2SortDims',
    #     '2SortDims+',
    #     '2LexSort',
    #     '2LexSort+',
    # ], rotation=0, fontsize=15, weight='bold')
    ax.set_xticklabels([
        'WC LowerBound',
        #
        # 'All-play-All',
        'Naive',
        #
        "SortedDims",  # "'2SortDims',
        # 'SortedDims+',
        'SortedDims++',
        "SingleDim",  # "'2LexSort',
        # 'SingleDim+',
        'SingleDim++',
    ],
        #    rotation=0, fontsize=5.5, weight='bold'
        # rotation=30, fontsize=9, weight='bold'
        rotation=30, fontsize='large', weight='bold'
    )

    #
    y_label_text = y_axis_label
    print("jkdfghil y_label_text", y_label_text)
    print("jkdfghil y_axis_label", y_axis_label)
    if y_axis_label == "num_comparisons":
        y_label_text = "Cost"
    if y_axis_label == "num_phases":
        y_label_text = "Latency"
    if y_axis_label == "skln_size":
        y_label_text = "Output Skyline Size"
    if y_axis_label == "Adapted_nDCG_at_GT_skln_size":
        y_label_text = "Adapted nDCG at Ground-Truth Skyline Size"
    if y_axis_label == "Adapted_nDCG_at_Algo_skln_size":
        y_label_text = "Adapted nDCG at Algorithm Skyline Size"
    #
    # plotter__HCOMP2020__whit_LowerBound
    #
    # {size in points, 'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
    # ax.set_ylabel(y_label_text, fontsize=15, weight='bold') SDM21
    ax.set_ylabel(y_label_text, fontsize=15, weight='bold')
    # ax.set_xlabel('Algorithm', fontsize=15, weight='bold')
    # ax.set_title(plot_title, size=15, weight='bold')
    #
    #
    #
    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='black', linewidth=2)

    for cap in bp['caps']:
        cap.set(color='black', linewidth=2)

    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='red', linewidth=4)

    ## change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        # change outline color
        box.set(color='black', linewidth=3)

    #
    #
    #
    #
    #
    #
    #
    #
    #
    plt.tight_layout()
    plt.savefig(complete_output_file_name)
    #

    return


def plotter__HCOMP2020__whit_LowerBound__ORIGINAL(map__algo__distrib,
                                                  plot_title, y_axis_label, complete_output_file_name):
    #
    # PLOT
    plt.figure(figsize=(6, 6))
    # sea.figure(figsize=(6, 6))

    # Create an axes instance
    # ax = plt.subplot(1, 1, 1)
    ax = plt.subplot()
    # ax = sea.subplot()

    lost__plot_titles_with_Ymin0_Ymin1 = ["precision",
                                          "recall",
                                          #
                                          "Adapted_nDCG_at_GT_skln_size",  # "nDCG@|GT_SKLN|",
                                          "Adapted_nDCG_at_Algo_skln_size",  # "nDCG@|Algo_SKLN|",
                                          ]
    #
    if plot_title in lost__plot_titles_with_Ymin0_Ymin1:
        plt.ylim(bottom=0, top=1)

    # plt.ylim(bottom=0, top=1)
    # plt.ylim(bottom=0)
    plt.grid(True, linestyle='-',
             color='lightgrey',
             alpha=0.5)

    bp = ax.boxplot([
        # map__algo__distrib["2SortDims"],
        # map__algo__distrib["2SortDims+"],
        # map__algo__distrib["2LexSort"],
        # map__algo__distrib["2LexSort+"],
        #
        map__algo__distrib["twoSort_skln NONE"],
        map__algo__distrib["twoSort_skln map__num_points__best_s"],
        map__algo__distrib["twoSort_skln map__num_points__best_s RPQS"],
        map__algo__distrib["skln__lex_sort_skln NONE"],
        map__algo__distrib["skln__lex_sort_skln map__num_points__best_s"],
        map__algo__distrib["skln__lex_sort_skln map__num_points__best_s RPQS"],
        map__algo__distrib["naive_method_for_skln"],
        map__algo__distrib["max_between_Lower_Bound_with_2_Max_Finding_and_Output_Sensitive_Lower_Bound"],
    ],
        widths=(0.4,
                0.4,
                0.4,

                0.4,
                0.4,
                0.4,

                0.4,

                0.4,
                ))
    #
    # ax.set_xticklabels([
    #     '2SortDims',
    #     '2SortDims+',
    #     '2LexSort',
    #     '2LexSort+',
    # ], rotation=0, fontsize=15, weight='bold')
    ax.set_xticklabels([
        "SortedDims",  # "'2SortDims',
        'SortedDims+',
        'SortedDims++',
        "SingleDim",  # "'2LexSort',
        'SingleDim+',
        'SingleDim++',
        'Naive',
        'WC LowerBound',
    ], rotation=30, fontsize=9, weight='bold')

    #
    ax.set_ylabel(y_axis_label, fontsize=15, weight='bold')
    ax.set_xlabel('Algorithm', fontsize=15, weight='bold')
    # ax.set_title(plot_title, size=15, weight='bold')
    #
    #
    #
    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='black', linewidth=2)

    for cap in bp['caps']:
        cap.set(color='black', linewidth=2)

    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='red', linewidth=4)

    ## change outline color, fill color and linewidth of the boxes
    for box in bp['boxes']:
        # change outline color
        box.set(color='black', linewidth=3)
    #
    plt.tight_layout()
    plt.savefig(complete_output_file_name)
    #

    return


#
##
###
def distribution_of_algorithm_output_characteristics_from_CROWD_EXPERIMENT_OUTPUT_FILE(input_file_name, algo_name):
    #
    map__feature_name__distribution = {}
    map__feature_name__distribution["num_comparisons"] = []
    map__feature_name__distribution["num_phases"] = []
    map__feature_name__distribution["skln_size"] = []
    #
    map__feature_name__distribution["precision"] = []
    map__feature_name__distribution["recall"] = []
    #
    map__feature_name__distribution["Adapted_nDCG_at_GT_skln_size"] = []
    map__feature_name__distribution["Adapted_nDCG_at_Algo_skln_size"] = []
    #
    # map__feature_name__distribution["max_between_Lower_Bound_with_2_Max_Finding_and_Output_Sensitive_Lower_Bound"] = []
    map__feature_name__distribution["num_comparisons_for_Output_Sensitive_Lower_Bound"] = []
    #
    #
    #
    input_file_handler = open(input_file_name, "r", 1000000)
    csv_reader = csv.reader(input_file_handler, delimiter='\t', quoting=csv.QUOTE_ALL)
    for record in csv_reader:
        c_algo_name = record[13]
        if c_algo_name != algo_name:
            continue
        #
        c_num_comparisons = int(record[14])
        c_num_phases = int(record[15])
        c_skln_size = int(record[16])
        #
        map__feature_name__distribution["num_comparisons"].append(c_num_comparisons)
        map__feature_name__distribution["num_phases"].append(c_num_phases)
        map__feature_name__distribution["skln_size"].append(c_skln_size)
        #
        #
        GT_skln_as_set__point_IDs = set(eval(record[12]))
        Algo_skln_as_set__point_IDs = set(eval(record[17]))
        map__metric_name__value = compute_precision_and_recall(Algo_skln_as_set__point_IDs, GT_skln_as_set__point_IDs)
        map__feature_name__distribution["precision"].append(map__metric_name__value["precision"])
        map__feature_name__distribution["recall"].append(map__metric_name__value["recall"])
        #
        #
        #
        GT_skln_size = len(GT_skln_as_set__point_IDs)
        distribution_of_Algo_skln_points_according_to_GT_skln_orders = list(eval(record[22]))
        # PATCH
        for index in range(len(distribution_of_Algo_skln_points_according_to_GT_skln_orders)):
            distribution_of_Algo_skln_points_according_to_GT_skln_orders[index] = 4 if \
                distribution_of_Algo_skln_points_according_to_GT_skln_orders[index] <= 0 else \
                distribution_of_Algo_skln_points_according_to_GT_skln_orders[index]
        Adapted_nDCG_at_GT_skln_size = compute_Adapted_nDCG_at_GT_skln_size(
            distribution_of_Algo_skln_points_according_to_GT_skln_orders, GT_skln_size)
        map__feature_name__distribution["Adapted_nDCG_at_GT_skln_size"].append(Adapted_nDCG_at_GT_skln_size)
        #
        GT__list_cuples_id__skln_order = eval(record[23])
        Adapted_nDCG_at_Algo_skln_size = compute_Adapted_nDCG_at_Algo_skln_size(
            distribution_of_Algo_skln_points_according_to_GT_skln_orders,
            GT__list_cuples_id__skln_order)
        map__feature_name__distribution["Adapted_nDCG_at_Algo_skln_size"].append(Adapted_nDCG_at_Algo_skln_size)
        #
        #
        num_comps_LB = int(record[8])
        map__feature_name__distribution["num_comparisons_for_Output_Sensitive_Lower_Bound"].append(num_comps_LB)
        # num_comps_LB = int(record[10])
        # map__feature_name__distribution["max_between_Lower_Bound_with_2_Max_Finding_and_Output_Sensitive_Lower_Bound"].append(num_comps_LB)
        #
    input_file_handler.close()
    return map__feature_name__distribution


#
##
###
def compute_complete_statistics(distribution_of_values):
    #
    num_samples = len(distribution_of_values)
    std = np.std(distribution_of_values)
    avg = sum(distribution_of_values) / num_samples
    min_val = min(distribution_of_values)
    q_1 = np.quantile(distribution_of_values, .25)
    median = np.quantile(distribution_of_values, .5)
    q_3 = np.quantile(distribution_of_values, .75)
    max_val = max(distribution_of_values)
    #
    avg = round(avg, 3)
    std = round(std, 3)
    min_val = round(min_val, 3)
    q_1 = round(q_1, 3)
    median = round(median, 3)
    q_3 = round(q_3, 3)
    max_val = round(max_val, 3)
    #
    map__name__value = {

        "num_samples": num_samples,

        "avg": avg,
        "std": std,

        "min": min_val,
        "q_1": q_1,
        "median": median,
        "q_3": q_3,
        "max": max_val,
    }
    sorted_list__name__value = [
        ("num_samples", map__name__value["num_samples"]),

        ("avg", map__name__value["avg"]),
        ("std", map__name__value["std"]),

        ("min", map__name__value["min"]),
        ("q_1", map__name__value["q_1"]),
        ("median", map__name__value["median"]),
        ("q_3", map__name__value["q_3"]),
        ("max", map__name__value["max"]),
    ]

    return map__name__value, sorted_list__name__value


def add__value_to_a_distribution_dictionary(map__feature__values_distribution, feature, value):
    if feature not in map__feature__values_distribution:
        map__feature__values_distribution[feature] = []
    map__feature__values_distribution[feature].append(value)
    return


def compute_distributions(input_file_name, number_of_features, all_feature_names):
    map__algo__n__feature__distribution = {}
    set__all_n_values = set()
    #
    input_file_handler = open(input_file_name, "r", 1000000)
    csv_reader = csv.reader(input_file_handler, delimiter='\t', quoting=csv.QUOTE_ALL)
    csv_reader.__next__()
    for record in csv_reader:
        c_num_dimensions = int(record[6])
        if c_num_dimensions != number_of_features:
            continue
        #
        #
        c_algo = record[18]
        if c_algo not in map__algo__n__feature__distribution:
            map__algo__n__feature__distribution[c_algo] = {}
        c_num_points = int(record[7])
        if c_num_points not in map__algo__n__feature__distribution[c_algo]:
            map__algo__n__feature__distribution[c_algo][c_num_points] = {}
        #
        #
        set__all_n_values.add(c_num_points)
        #
        #
        c_map__feature__values_distribution = map__algo__n__feature__distribution[c_algo][c_num_points]
        #
        #
        c_num_comparisons = int(record[19])
        add__value_to_a_distribution_dictionary(c_map__feature__values_distribution, "num_comparisons",
                                                c_num_comparisons)
        c_num_phases = int(record[20])
        add__value_to_a_distribution_dictionary(c_map__feature__values_distribution, "num_phases",
                                                c_num_phases)
        c_skln_size = int(record[21])
        add__value_to_a_distribution_dictionary(c_map__feature__values_distribution, "skln_size",
                                                c_skln_size)
        #
        #
        GT_skln_as_set__point_IDs = set(eval(record[17]))
        Algo_skln_as_set__point_IDs = set(eval(record[23]))
        map__metric_name__value = compute_precision_and_recall(Algo_skln_as_set__point_IDs, GT_skln_as_set__point_IDs)
        c_precision = map__metric_name__value["precision"]
        add__value_to_a_distribution_dictionary(c_map__feature__values_distribution, "precision",
                                                c_precision)
        c_recall = map__metric_name__value["recall"]
        add__value_to_a_distribution_dictionary(c_map__feature__values_distribution, "recall",
                                                c_recall)
        #
        #
        GT_skln_size = len(GT_skln_as_set__point_IDs)
        distribution_of_Algo_skln_points_according_to_GT_skln_orders = list(eval(record[28]))
        c_Adapted_nDCG_at_GT_skln_size = compute_Adapted_nDCG_at_GT_skln_size(
            distribution_of_Algo_skln_points_according_to_GT_skln_orders, GT_skln_size)
        add__value_to_a_distribution_dictionary(c_map__feature__values_distribution, "Adapted_nDCG_at_GT_skln_size",
                                                c_Adapted_nDCG_at_GT_skln_size)
        #
        #
        GT__list_cuples_id__skln_order = eval(record[33])
        c_Adapted_nDCG_at_Algo_skln_size = compute_Adapted_nDCG_at_Algo_skln_size(
            distribution_of_Algo_skln_points_according_to_GT_skln_orders,
            GT__list_cuples_id__skln_order)
        add__value_to_a_distribution_dictionary(c_map__feature__values_distribution, "Adapted_nDCG_at_Algo_skln_size",
                                                c_Adapted_nDCG_at_Algo_skln_size)
        #
        #
        c_num_comps_LB = int(record[14])
        add__value_to_a_distribution_dictionary(c_map__feature__values_distribution,
                                                "max_between_Lower_Bound_with_2_Max_Finding_and_Output_Sensitive_Lower_Bound",
                                                c_num_comps_LB)
        c_GT_skln_size = int(record[15])
        add__value_to_a_distribution_dictionary(c_map__feature__values_distribution, "GT_skln_SIZE",
                                                c_GT_skln_size)
        #
        #
        ################
        ### ACCURACY ###
        c_sorted_distribution_of_distances_of_Algo_skln_points_from_the_GT_skln_FRONTIER = eval(record[29])
        c_sorted_distribution_of_distances_of_Algo_skln_points_from_the_GT_skln_FRONTIER.sort()
        # nnnn = len(c_sorted_distribution_of_distances_of_Algo_skln_points_from_the_GT_skln_FRONTIER)
        # c_min_distance_from_GT_skln_error = c_sorted_distribution_of_distances_of_Algo_skln_points_from_the_GT_skln_FRONTIER[0]
        # c_median_distance_from_GT_skln_error = c_sorted_distribution_of_distances_of_Algo_skln_points_from_the_GT_skln_FRONTIER[int(nnnn / 2)]
        c_max_distance_from_GT_skln_error = \
            c_sorted_distribution_of_distances_of_Algo_skln_points_from_the_GT_skln_FRONTIER[-1]
        # add__value_to_a_distribution_dictionary(c_map__feature__values_distribution,
        #                                        "min_distance_from_GT_skln_error",
        #                                        c_min_distance_from_GT_skln_error)
        # add__value_to_a_distribution_dictionary(c_map__feature__values_distribution,
        #                                        "median_distance_from_GT_skln_error",
        #                                        c_median_distance_from_GT_skln_error)
        # add__value_to_a_distribution_dictionary(c_map__feature__values_distribution,
        #                                        "max_distance_from_GT_skln_error",
        #                                        c_max_distance_from_GT_skln_error)
        add__value_to_a_distribution_dictionary(c_map__feature__values_distribution,
                                                "Accuracy",
                                                c_max_distance_from_GT_skln_error)
        #
        #
        #
        ##################
        ### CORRECNESS ###
        # print("kghjgugi", record[31])
        sorted_distribution_CORRECTNESS_values = []
        if "inf" in str(record[31]):
            sorted_distribution_CORRECTNESS_values = [1000] * (str(record[31]).count(",") + 1)
        else:
            sorted_distribution_CORRECTNESS_values = eval(record[31])
        # print("sorted_distribution_CORRECTNESS_values", sorted_distribution_CORRECTNESS_values)
        # print("bcvxgd", sorted_distribution_CORRECTNESS_values)
        sorted_distribution_CORRECTNESS_values.sort()
        c_max__CORRECTNESS_value = sorted_distribution_CORRECTNESS_values[-1]
        add__value_to_a_distribution_dictionary(c_map__feature__values_distribution,
                                                "Correctness",
                                                c_max__CORRECTNESS_value)
        #
        #
        #
        #
        #
        #
    input_file_handler.close()
    #
    sulist_all_n_values = sorted(list(set__all_n_values))
    return map__algo__n__feature__distribution, sulist_all_n_values


##########################################################################################################
##########################################################################################################

def statistics_for_EXP_CROWD():
    # input_file_name = "./output/output_SKLN_EXP__CROWD__2020_06_03__10_26_59_781266.tsv"
    # input_file_name = "./output/output_SKLN_EXP__CROWD__2020_06_04__18_50_29_441193.tsv"
    # input_file_name = "./output/output_SKLN_EXP__CROWD__2020_06_06__17_00_33_761202.tsv"
    #
    #
    ################
    ### HCOMP20 ###
    # input_file_name = "./output/definitive/DEFINITIVE__output_SKLN_EXP__CROWD__2020_06_07__18_30_59_352342.tsv"
    #
    #
    ##############
    ### SDM21 ###
    # input_file_name = "./output/output_SKLN_EXP__CROWD__2020_09_07__16_30_00_583703.tsv"  ## 10 samples
    # input_file_name = "./output/output_SKLN_EXP__CROWD__2020_09_07__17_03_31_217557.tsv"  ## 1000 samples
    #
    # input_file_name = "./output/output_SKLN_EXP__CROWD__2020_09_20__16_13_07_767657.tsv"; dataset_name = "EXP_CROWD__AVOIDING_TIES"  ## 100 samples extract_already_performed_comparisons(
    # input_file_name = "./output/output_SKLN_EXP__CROWD__2020_09_20__16_18_46_205110.tsv"; dataset_name = "EXP_CROWD__WITH_TIES"  ## 100 samples extract_already_performed_comparisons_RANDOMLY(
    #
    #
    # input_file_name = "./output/output_SKLN_EXP__CROWD__2020_09_20__16_41_50_247218.tsv";
    # dataset_name = "EXP_CROWD__AVOIDING_TIES_012_"
    #
    # input_file_name = "./output/output_SKLN_EXP__CROWD__2020_09_20__16_43_41_049917.tsv";
    # dataset_name = "EXP_CROWD__AVOIDING_TIES_01_"
    #
    # input_file_name = "./output/output_SKLN_EXP__CROWD__2020_09_20__16_45_56_679356.tsv";
    # dataset_name = "EXP_CROWD__AVOIDING_TIES_02_"
    #
    # input_file_name = "./output/output_SKLN_EXP__CROWD__2020_09_20__16_47_40_247626.tsv";
    # dataset_name = "EXP_CROWD__AVOIDING_TIES_12_" ### SHITTY PLOTS!!! Due to a single point in skln and all the other n-1 in level 2 :\
    #
    #
    ### SDM21 ###
    # input_file_name = "./output/output_SKLN_EXP__CROWD__2020_09_20__16_50_16_080208.tsv";
    # dataset_name = "EXP_CROWD__1000samples__AVOIDING_TIES__features_012__fliersON_"
    #
    ## with ApA 100 samples
    # input_file_name = "./output/output_SKLN_EXP__CROWD__2020_10_06__10_55_15_800974.tsv";  ### 100 samples
    # dataset_name = "EXP_CROWD__100samples__AVOIDING_TIES__features_012__fliersON__withAPA_"
    ## with ApA 1000 samples
    # input_file_name = "./output/output_SKLN_EXP__CROWD__2020_10_06__20_17_51_386995.tsv";  ### 1000 samples
    dataset_name = "EXP_CROWD__1000samples__AVOIDING_TIES__features_012__fliersON__withAPA_"
    ## without ApA 1000 samples
    input_file_name = "./output/output_SKLN_EXP__CROWD__2020_10_06__20_17_51_386995.tsv";
    # dataset_name = "EXP_CROWD__1000samples__AVOIDING_TIES__features_012__fliersON_"

    #
    #
    all_algo_names = [
        "naive_method_for_skln",
        "twoSort_skln NONE",
        # "twoSort_skln map__num_points__best_s",
        "twoSort_skln map__num_points__best_s RPQS",
        "skln__lex_sort_skln NONE",
        # "skln__lex_sort_skln map__num_points__best_s",
        "skln__lex_sort_skln map__num_points__best_s RPQS",
        #
        "ApA",
        # "ApA complete comparisons",
    ]

    print()
    print()

    all_feature_names = [
        "num_comparisons",
        "num_phases",
        "skln_size",
        #
        # "precision",
        # "recall",
        #
        "Adapted_nDCG_at_GT_skln_size",  # "nDCG@|GT_SKLN|",
        "Adapted_nDCG_at_Algo_skln_size",  # "nDCG@|Algo_SKLN|",
        #
        # "max_between_Lower_Bound_with_2_Max_Finding_and_Output_Sensitive_Lower_Bound",
        "num_comparisons_for_Output_Sensitive_Lower_Bound",
    ]

    map__algo__algorithm_feature_name__distribution = {}

    for c_feature_name in all_feature_names:
        for algo_name in all_algo_names:
            #
            if algo_name not in map__algo__algorithm_feature_name__distribution:
                map__algo__algorithm_feature_name__distribution[algo_name] = {}
            # if c_feature_name not in map__algo__algorithm_feature_name__distribution[algo_name]:
            #    map__algo__algorithm_feature_name__distribution[algo_name][c_feature_name]
            #
            map__algorithm_feature_name__distribution = distribution_of_algorithm_output_characteristics_from_CROWD_EXPERIMENT_OUTPUT_FILE(
                input_file_name,
                algo_name)
            map__algo__algorithm_feature_name__distribution[algo_name] = map__algorithm_feature_name__distribution
            #
            distribution = map__algorithm_feature_name__distribution[c_feature_name]
            stats_as__map__name__value, stats_as__sorted_list__name__value = compute_complete_statistics(distribution)
            #
            print("feature_name:", c_feature_name)
            print("algo_name:", algo_name)
            # pp.pprint(stats)
            pp.pprint(stats_as__sorted_list__name__value)
            print()
            print()

    # Box-Plot: for each algorithm distribution of {#comp, }
    for c_feature_name in all_feature_names[:-1]:
        #
        map__algo__distrib = {}
        for alg in map__algo__algorithm_feature_name__distribution:
            if alg not in map__algo__distrib:
                map__algo__distrib[alg] = []
            map__algo__distrib[alg] = map__algo__algorithm_feature_name__distribution[alg][c_feature_name]
        #
        #
        #
        # if c_feature_name == "num_comparisons":
        #    map__algo__distrib["max_between_Lower_Bound_with_2_Max_Finding_and_Output_Sensitive_Lower_Bound"] = \
        #        map__algo__algorithm_feature_name__distribution["twoSort_skln map__num_points__best_s"][
        #            "max_between_Lower_Bound_with_2_Max_Finding_and_Output_Sensitive_Lower_Bound"]
        if c_feature_name == "num_comparisons":
            map__algo__distrib["num_comparisons_for_Output_Sensitive_Lower_Bound"] = \
                map__algo__algorithm_feature_name__distribution["twoSort_skln NONE"][
                    "num_comparisons_for_Output_Sensitive_Lower_Bound"]
        #
        #
        plot_title = c_feature_name
        y_axis_label = c_feature_name
        complete_output_file_name = "./output/BoxPlot__" + dataset_name + "__" + plot_title + ".pdf"
        if c_feature_name == "num_comparisons":
            plotter__HCOMP2020__whit_LowerBound(map__algo__distrib, plot_title, y_axis_label, complete_output_file_name)
        else:
            plotter__HCOMP2020(map__algo__distrib, plot_title, y_axis_label, complete_output_file_name)
    return


def plot(plot_title, map__algo___x_y_error, plot_file_name, map__algo__plot_feature__value, input_feature):
    #
    plt.figure(figsize=(7, 4))
    ax = plt.subplot(1, 1, 1)
    #
    #
    for c_algo in map__algo___x_y_error:
        x = np.array([x for x, y, error in map__algo___x_y_error[c_algo]])
        y = np.array([y for x, y, error in map__algo___x_y_error[c_algo]])
        error = np.array([error for x, y, error in map__algo___x_y_error[c_algo]])
        #
        # ax.plot(x, y, 'black', linestyle='-', linewidth=1, alpha=1., label=c_algo, marker='2',
        #        markersize=4, markeredgewidth=2)
        # ax.fill_between(x, y + error, y - error, alpha=0.1, facecolor='black')
        ax.plot(x, y,
                map__algo__plot_feature__value[c_algo]["color"],
                linestyle=map__algo__plot_feature__value[c_algo]["linestyle"],
                linewidth=2, alpha=1,
                label=map__algo__plot_feature__value[c_algo]["name"],
                marker=map__algo__plot_feature__value[c_algo]["marker"],
                markersize=8, markeredgewidth=2,
                markerfacecolor='none', markeredgecolor=map__algo__plot_feature__value[c_algo]["color"])
        ax.fill_between(x, y + error, y - error, alpha=0.125, facecolor=map__algo__plot_feature__value[c_algo]["color"])
    #
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    #
    y_label_text = input_feature
    if input_feature == "Correctness":
        y_label_text = "Covering"
    if input_feature == "Accuracy":
        y_label_text = "Succinctness"
    if input_feature == "num_phases":
        y_label_text = "Latency"
    if input_feature == "num_comparisons":
        y_label_text = "Cost"
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    if input_feature == "skln_size":
        y_label_text = "Output Skyline Size"
    if input_feature == "Adapted_nDCG_at_GT_skln_size":
        y_label_text = "Adapted nDCG at Ground-Truth Skyline Size"
    if input_feature == "Adapted_nDCG_at_Algo_skln_size":
        y_label_text = "Adapted nDCG at Algorithm Skyline Size"
    #
    # {size in points, 'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
    # ax.set_ylabel(y_label_text, size='large', weight='bold')
    # ax.set_xlabel("n", size='large', weight='bold')
    ax.set_ylabel(y_label_text, size='xx-large', weight='bold')
    ax.set_xlabel("n", size='xx-large', weight='bold')
    #
    # ax.set_xticklabels(x_ticks, rotation=0, fontsize=8)
    # ax.set_yticklabels(x_ticks, rotation=0, fontsize=8)
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')
    #
    #
    #############
    ## LEGEND ###
    handles, labels = ax.get_legend_handles_labels()
    # new_handles = [handles[6], handles[5], handles[4], handles[3], handles[2], handles[1], handles[0]]
    # new_labels = [labels[6], labels[5], labels[4], labels[3], labels[2], labels[1], labels[0]]
    #
    # new_handles = [handles[4], handles[3], handles[2], handles[1], handles[0]]
    # new_labels = [labels[4], labels[3], labels[2], labels[1], labels[0]]
    #
    #
    #
    #
    # for index, content in enumerate(handles):
    #     print("handles", index, content)
    # for index, content in enumerate(labels):
    #     print("labels", index, content)
    # print()
    #
    #
    #
    if input_feature == "num_comparisons" or input_feature == "skln_size":
        # new_handles = [handles[7], handles[6], handles[5], handles[4], handles[3], handles[2], handles[1], handles[0]]
        # new_labels = [labels[7], labels[6], labels[5], labels[4], labels[3], labels[2], labels[1], labels[0]]
        #
        # new_handles = [handles[5], handles[0], handles[1], handles[2], handles[3], handles[4], ]
        # new_labels = [labels[5], labels[0], labels[1], labels[2], labels[3], labels[4], ]
        new_handles = [handles[1], handles[2], handles[3], handles[4], handles[0], handles[5]]
        new_labels = [labels[1], labels[2], labels[3], labels[4], labels[0], labels[5], ]
        #
        # ax.legend(new_handles, new_labels, fontsize='large', loc="best", ncol=2)
        # ax.legend(new_handles, new_labels, fontsize='large', loc="best", ncol=3, prop={'weight': 'bold', 'size': "large"}) # Cool one ;)
        ax.legend(new_handles, new_labels, fontsize='large', loc="best", ncol=1,
                  prop={'weight': 'bold', 'size': "large"})
    else:
        new_handles = [handles[1], handles[2], handles[3], handles[4], handles[0]]
        new_labels = [labels[1], labels[2], labels[3], labels[4], labels[0]]
        # new_handles = handles
        # new_labels = labels
        ax.legend(new_handles, new_labels, fontsize='large', loc="best", ncol=3,
                  prop={'weight': 'bold', 'size': "large"})
    # ax.legend(new_handles, new_labels, fontsize='large', loc="upper left", ncol=1)
    # ax.legend(new_handles, new_labels, fontsize='large', loc="best", ncol=2)
    if (plot_title == "qqqqqqq"):
        ax.legend(new_handles, new_labels, fontsize='large', loc="center right", ncol=1)
    if (plot_title == "qqqq"):
        ax.legend(new_handles, new_labels, fontsize='large', loc="best", ncol=2)

    ax.grid()

    list__input_features_with_Ymin0_Ymax1 = ["precision",
                                             "recall",
                                             #
                                             "Adapted_nDCG_at_GT_skln_size",  # "nDCG@|GT_SKLN|",
                                             "Adapted_nDCG_at_Algo_skln_size",  # "nDCG@|Algo_SKLN|",
                                             ]
    #
    if input_feature in list__input_features_with_Ymin0_Ymax1:
        plt.ylim(bottom=0, top=1)
    list__input_features_with_Ymin0_Ymax2 = ["median_distance_from_GT_skln_error",
                                             "max_distance_from_GT_skln_error",
                                             "Accuracy",
                                             "Correctness",
                                             "min_distance_from_GT_skln_error", ]
    if input_feature in list__input_features_with_Ymin0_Ymax2:
        plt.ylim(bottom=0, top=2)
    if input_feature == "Correctness":
        # pass
        plt.ylim(bottom=0, top=0.4)

    #
    # attempts to change the y labels:
    # plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    # y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
    # ax.yaxis.set_major_formatter(y_formatter)
    # plt.autoscale(enable=True)
    # ax.autoscale(enable=True)
    plt.tight_layout()
    plt.savefig(plot_file_name)
    plt.clf()
    return


#
##
###
def plot_2(plot_title, map__algo___x_y, plot_file_name, map__algo__plot_feature__value, input_feature):
    #
    plt.figure(figsize=(7, 4))
    ax = plt.subplot(1, 1, 1)
    #
    #
    for c_algo in map__algo___x_y:
        x = np.array([x for x, y in map__algo___x_y[c_algo]])
        y = np.array([y for x, y in map__algo___x_y[c_algo]])
        #
        # ax.plot(x, y, 'black', linestyle='-', linewidth=1, alpha=1., label=c_algo, marker='2',
        #        markersize=4, markeredgewidth=2)
        # ax.fill_between(x, y + error, y - error, alpha=0.1, facecolor='black')
        ax.plot(x, y,
                map__algo__plot_feature__value[c_algo]["color"],
                linestyle=map__algo__plot_feature__value[c_algo]["linestyle"],
                linewidth=2, alpha=1,
                label=map__algo__plot_feature__value[c_algo]["name"],
                marker=map__algo__plot_feature__value[c_algo]["marker"],
                markersize=8, markeredgewidth=2,
                markerfacecolor='none', markeredgecolor=map__algo__plot_feature__value[c_algo]["color"])
    #
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel(input_feature, size='large', weight='bold')
    ax.set_xlabel("n", size='large', weight='bold')
    #############
    ## LEGEND ###
    handles, labels = ax.get_legend_handles_labels()
    new_handles = [handles[6], handles[5], handles[4], handles[3], handles[2], handles[1], handles[0]]
    new_labels = [labels[6], labels[5], labels[4], labels[3], labels[2], labels[1], labels[0]]
    if input_feature == "num_comparisons" or input_feature == "skln_size":
        new_handles = [handles[7], handles[6], handles[5], handles[4], handles[3], handles[2], handles[1], handles[0]]
        new_labels = [labels[7], labels[6], labels[5], labels[4], labels[3], labels[2], labels[1], labels[0]]
    # ax.legend(new_handles, new_labels, fontsize='large', loc="upper left", ncol=1)
    ax.legend(new_handles, new_labels, fontsize='large', loc="best", ncol=1)
    if (plot_title == "qqqqqqq"):
        ax.legend(new_handles, new_labels, fontsize='large', loc="center right", ncol=1)
    if (plot_title == "qqqq"):
        ax.legend(new_handles, new_labels, fontsize='large', loc="best", ncol=2)

    ax.grid()

    list__input_features_with_Ymin0_Ymax1 = ["precision",
                                             "recall",
                                             #
                                             "Adapted_nDCG_at_GT_skln_size",  # "nDCG@|GT_SKLN|",
                                             "Adapted_nDCG_at_Algo_skln_size",  # "nDCG@|Algo_SKLN|",
                                             ]
    #
    if input_feature in list__input_features_with_Ymin0_Ymax1:
        plt.ylim(bottom=0, top=1)
    list__input_features_with_Ymin0_Ymax2 = ["median_distance_from_GT_skln_error",
                                             "max_distance_from_GT_skln_error",
                                             "Accuracy",
                                             "Correctness",
                                             "min_distance_from_GT_skln_error", ]
    if input_feature in list__input_features_with_Ymin0_Ymax2:
        plt.ylim(bottom=0, top=2)

    #
    # attempts to change the y labels:
    # plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    # y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
    # ax.yaxis.set_major_formatter(y_formatter)
    # plt.autoscale(enable=True)
    # ax.autoscale(enable=True)
    plt.tight_layout()
    plt.savefig(plot_file_name)
    plt.clf()
    return


########################################################################################################################
########################################################################################################################

# statistics_for_EXP_CROWD()
# exit(0)

#
#
#
#
# prefix_output_file_names = "RANDOM__STANDARD_WORKER__plot_error_area__"
# input_file_name = "/Users/ikki/temp/skln__definitive_output_exps_Random_and_OLM/output__worker_policy_standard/output_SKLN_EXP__RANDOM__worker_policy_standard__DEFINITIVE.tsv"
#
# prefix_output_file_names = "SIM_REAL__STANDARD_WORKER__plot_error_area__"
# input_file_name = "/Users/ikki/temp/skln__definitive_output_exps_Random_and_OLM/output__worker_policy_standard/output_SKLN_EXP__SIMULATION_REAL_DATA__worker_policy_standard__DEFINITIVE.tsv"
#
#
#
########## prefix_output_file_names = "RANDOM__ALWAYS_WRONG_WORKER__plot_error_area__"
########## input_file_name = "/Users/ikki/temp/skln__definitive_output_exps_Random_and_OLM/output__worker_policy_always_wrong/output_SKLN_EXP__RANDOM__worker_policy_always_wrong__DEFINITIVE.tsv"
#
########## prefix_output_file_names = "SIM_REAL__ALWAYS_WRONG_WORKER__plot_error_area__"
########## input_file_name = "/Users/ikki/temp/skln__definitive_output_exps_Random_and_OLM/output__worker_policy_always_wrong/output_SKLN_EXP__SIMULATION_REAL_DATA__worker_policy_always_wrong__DEFINITIVE.tsv"

#
#
######### prefix_output_file_names = "SIM_REAL_DATA_HOUSING__01__STANDARD_WORKER__plot_error_area__"
########## input_file_name = "/Users/ikki/temp/DATASET__housing/output_SKLN_EXP__REAL_DATA_HOUSING__delta_01/output_SKLN_EXP__REAL_DATA_HOUSING__worker_policy_standard__delta_01__.tsv"
#
########## prefix_output_file_names = "SIM_REAL_DATA_HOUSING__01__ALWAYS_WRONG_WORKER__plot_error_area__"
########## input_file_name = "/Users/ikki/temp/DATASET__housing/output_SKLN_EXP__REAL_DATA_HOUSING__delta_01/output_SKLN_EXP__REAL_DATA_HOUSING__worker_policy_always_wrong__delta_01__.tsv"

#
#
# prefix_output_file_names = "SIM_REAL_DATA_HOUSING__005__STANDARD_WORKER__plot_error_area__"
# input_file_name = "/Users/ikki/temp/DATASET__housing/output_SKLN_EXP__REAL_DATA_HOUSING__delta_005/output_SKLN_EXP__REAL_DATA_HOUSING__worker_policy_standard__delta_005__.tsv"
#
#####prefix_output_file_names = "SIM_REAL_DATA_HOUSING__005__ALWAYS_WRONG_WORKER__plot_error_area__"
####input_file_name = "/Users/ikki/temp/DATASET__housing/output_SKLN_EXP__REAL_DATA_HOUSING__delta_005/output_SKLN_EXP__REAL_DATA_HOUSING__worker_policy_always_wrong__delta_005__.tsv"


#
######################
### TEST AND DEBUG ###
######################
#
prefix_output_file_names = "TEST__"
# input_file_name = "/Users/ikki/Dropbox/SKLN_in_CS/sw/Skyline/output/output_SKLN_EXP__RANDOM__worker_policy_standard__2020_06_27__16_48_49_062375.tsv"
# input_file_name = "/Users/ikki/Dropbox/SKLN_in_CS/sw/Skyline/output/output_SKLN_EXP__RANDOM__worker_policy_standard__2020_08_10__10_43_31_958166.tsv"

### SortedDims with different values for parameter s in 2Sort...
# input_file_name = "/Users/ikki/Dropbox/SKLN_in_CS/sw/Skyline/output/output_SKLN_EXP__RANDOM__worker_policy_standard__2020_08_10__11_03_25_128700.tsv"
# input_file_name = "/Users/ikki/Dropbox/SKLN_in_CS/sw/Skyline/output/output_SKLN_EXP__RANDOM__worker_policy_standard__2020_08_11__18_48_46_827564.tsv"
# input_file_name = "/Users/ikki/Dropbox/SKLN_in_CS/sw/Skyline/output/output_SKLN_EXP__RANDOM__worker_policy_standard__2020_08_11__19_12_57_713663.tsv"
# input_file_name = "/Users/ikki/Dropbox/SKLN_in_CS/sw/Skyline/output/output_SKLN_EXP__RANDOM__worker_policy_standard__2020_08_12__07_30_38_803791.tsv"

### SingleDims with different values for parameter s in 2Sort...
# input_file_name = "/Users/ikki/Dropbox/SKLN_in_CS/sw/Skyline/output/output_SKLN_EXP__RANDOM__worker_policy_standard__2020_08_12__08_13_10_228476.tsv"

### SortedDims with different values for parameter s and APA_2.
# input_file_name = "/Users/ikki/Dropbox/SKLN_in_CS/sw/Skyline/output/output_SKLN_EXP__RANDOM__worker_policy_standard__2020_08_14__08_02_13_104430.tsv"
# input_file_name = "/Users/ikki/Dropbox/SKLN_in_CS/sw/Skyline/output/output_SKLN_EXP__RANDOM__worker_policy_standard__2020_08_14__08_05_37_338984.tsv"

### SortedDims with different values for parameter s and APA_3.
# input_file_name = "/Users/ikki/Dropbox/SKLN_in_CS/sw/Skyline/output/output_SKLN_EXP__RANDOM__worker_policy_standard__2020_08_24__09_12_43_213791.tsv"
# Always Wrong Workers.
# input_file_name = "/Users/ikki/Dropbox/SKLN_in_CS/sw/Skyline/output/output_SKLN_EXP__RANDOM__worker_policy_always_wrong__2020_08_24__11_53_02_892953.tsv"


################################################################################################
################################################################################################
################################################################################################
### Sample for SDM21
### Random points random-workers
# prefix_output_file_names = "ra__TEST__"
# input_file_name = "/Users/ikki/Dropbox/SKLN_in_CS/sw/Skyline/output/output_SKLN_EXP__RANDOM__worker_policy_standard__2020_09_07__10_32_02_208705.tsv"
##input_file_name = "/Users/ikki/Dropbox/SKLN_in_CS/sw/Skyline/output/output_SKLN_EXP__RANDOM__worker_policy_standard__2020_09_09__19_04_45_561353.tsv"
# input_file_name = "/Users/ikki/Dropbox/SKLN_in_CS/sw/Skyline/output/output_SKLN_EXP__RANDOM__worker_policy_standard__2020_09_09__19_36_34_266750.tsv"
# input_file_name = "/Users/ikki/Dropbox/SKLN_in_CS/sw/Skyline/output/output_SKLN_EXP__RANDOM__worker_policy_standard__2020_09_09__19_44_44_676021.tsv"

### Random points always-wrong-workers
# prefix_output_file_names = "aw__TEST__"
# input_file_name = "/Users/ikki/Dropbox/SKLN_in_CS/sw/Skyline/output/output_SKLN_EXP__RANDOM__worker_policy_always_wrong__2020_09_07__10_29_12_456017.tsv"

### housing random-workers
# prefix_output_file_names = "ra__TEST__HOUSING__"
# input_file_name = "/Users/ikki/Dropbox/SKLN_in_CS/sw/Skyline/output/output_SKLN_EXP__REAL_DATA_HOUSING__worker_policy_standard__2020_09_07__15_48_58_149855.tsv"# delta = 0.05
### housing always-wrong-workers
# prefix_output_file_names = "aw__TEST__HOUSING__"
# input_file_name = "/Users/ikki/Dropbox/SKLN_in_CS/sw/Skyline/output/output_SKLN_EXP__REAL_DATA_HOUSING__worker_policy_always_wrong__2020_09_07__15_51_56_375741.tsv"  # delta = 0.05
#
###prefix_output_file_names = "ra__TEST__HOUSING__"#
###input_file_name = "/Users/ikki/Dropbox/SKLN_in_CS/sw/Skyline/output/output_SKLN_EXP__REAL_DATA_HOUSING__worker_policy_standard__2020_09_07__15_30_26_355471.tsv" # delta = 0.1
#### housing always-wrong-workers
####prefix_output_file_names = "aw__TEST__HOUSING__"
###input_file_name = "/Users/ikki/Dropbox/SKLN_in_CS/sw/Skyline/output/output_SKLN_EXP__REAL_DATA_HOUSING__worker_policy_always_wrong__2020_09_07__15_36_26_597148.tsv"# delta = 0.1


#
#
#
#
#
# all_features_numbers = [2, 3, 4]
# all_features_numbers = [2, 3, 4, 5, 6]
# all_features_numbers = [4]
# all_features_numbers = [2]

################################################
### DEFINITIVE SDM21 ###########################
################################################
################################################
### DEFINITIVE CIKM21 ##########################
################################################
# prefix_output_file_names = "ra__RANDOM__"  ### SELECTED FOR CIKM21
## input_file_name = "/Users/ikki/Dropbox/SKLN_in_CS/sw/Skyline/output/SDM21/output_SKLN_EXP__RANDOM__worker_policy_standard__100_repetitions__2_3_4_dimensions.tsv"  # delta = 0.1
# input_file_name = "/Users/ikki/Dropbox/SKLN_in_CS/sw/Skyline/output/SDM21/output_SKLN_EXP__RANDOM__worker_policy_standard__100_repetitions__2_dimensions.tsv"  # delta = 0.1
# all_features_numbers = [2]
### all_features_numbers = [2, 3, 4]
#
prefix_output_file_names = "ra__HOUSING__"  ### SELECTED FOR CIKM21
#### input_file_name = "/Users/ikki/Dropbox/SKLN_in_CS/sw/Skyline/output/SDM21/output_SKLN_EXP__REAL_DATA_HOUSING__worker_policy_standard__100_repetitions__2_3_4_dimensions.tsv"  # delta = 0.05
input_file_name = "/Users/ikki/Dropbox/SKLN_in_CS/sw/Skyline/output/SDM21/output_SKLN_EXP__REAL_DATA_HOUSING__worker_policy_standard__100_repetitions__4_dimensions.tsv"  # delta = 0.05
all_features_numbers = [4]
################################################
################################################
################################################
#
###################
# prefix_output_file_names = "aww__HOUSING__"
# input_file_name = "/Users/ikki/Dropbox/SKLN_in_CS/sw/Skyline/output/SDM21/output_SKLN_EXP__REAL_DATA_HOUSING__worker_policy_AWW__100_repetitions__2_3_4_dimensions.tsv"  # delta = 0.05
# all_features_numbers = [2, 3, 4]
#
# prefix_output_file_names = "aww__RANDOM__"
# input_file_name = "/Users/ikki/Dropbox/SKLN_in_CS/sw/Skyline/output/SDM21/output_SKLN_EXP__RANDOM__worker_policy_AWW__100_repetitions__2_3_4_dimensions.tsv"  # delta = 0.1
##################

######### TO AVOID ################################
######### prefix_output_file_names = "ra__HOUSING_wApA__"
######### input_file_name = "/Users/ikki/Dropbox/SKLN_in_CS/sw/Skyline/output/output_SKLN_EXP__REAL_DATA_HOUSING__worker_policy_standard__with_ApA.tsv"  # delta = 0.05
######### input_file_name = "/Users/ikki/Dropbox/SKLN_in_CS/sw/Skyline/output/TEST.tsv"  # delta = 0.05
######### all_features_numbers = [2, 3, 4]

# all_features_numbers = [2, 3, 4]
# all_features_numbers = [2]
# all_features_numbers = [4]

### TEST #########
# prefix_output_file_names = "AAATEST__ra__HOUSING__"
# input_file_name = "/Users/ikki/Dropbox/SKLN_in_CS/sw/Skyline/TEST.tsv"
# all_features_numbers = [2, 3, 4]


################################################
################################################
################################################
all_algo_names = [
    "naive_method_for_skln",
    "twoSort_skln NONE",
    # "twoSort_skln map__num_points__best_s",
    "twoSort_skln map__num_points__best_s RPQS",
    "skln__lex_sort_skln NONE",
    # "skln__lex_sort_skln map__num_points__best_s",
    "skln__lex_sort_skln map__num_points__best_s RPQS",

    # "ApA",
]

print()
print()

all_feature_names = [
    "num_comparisons",
    "num_phases",
    "skln_size",
    #
    # "precision",
    # "recall",
    #
    # "Adapted_nDCG_at_GT_skln_size",  # "nDCG@|GT_SKLN|",
    # "Adapted_nDCG_at_Algo_skln_size",  # "nDCG@|Algo_SKLN|",
    #
    ###### "max_between_Lower_Bound_with_2_Max_Finding_and_Output_Sensitive_Lower_Bound",
    #
    ###### "min_distance_from_GT_skln_error",
    ###### "median_distance_from_GT_skln_error",
    ###### "max_distance_from_GT_skln_error",
    #
    "Accuracy",
    "Correctness",
]

map__algo__plot_feature__value = {
    "ApA": {
        "color": 'orange',
        "linestyle": 'solid',
        "marker": 'D',
        "name": "ApA",
    },

    "naive_method_for_skln": {
        "color": 'blue',
        "linestyle": 'solid',
        "marker": 'v',
        "name": "Naive",
    },
    #
    "skln__lex_sort_skln map__num_points__best_s RPQS": {
        # "color": 'red',
        "color": 'darkgreen',
        "linestyle": 'dashed',
        "marker": '1',
        "name": "SingleDim++",
    },
    "skln__lex_sort_skln map__num_points__best_s": {
        "color": 'green',
        "linestyle": 'dashed',
        "marker": 'x',
        "name": "SingleDim+",
    },
    "skln__lex_sort_skln NONE": {
        # "color": 'orange',
        # "color": '#009600',
        "color": 'black',
        "linestyle": 'dashed',
        "marker": 's',
        "name": "SingleDim",
    },
    #
    "twoSort_skln map__num_points__best_s RPQS": {
        "color": 'red',
        # "color": 'darkgreen',
        "linestyle": 'dotted',
        "marker": '2',
        "name": "SortedDims++",
    },
    "twoSort_skln map__num_points__best_s": {
        "color": 'green',
        "linestyle": 'dotted',
        "marker": '+',
        "name": "SortedDims+",
    },
    "twoSort_skln NONE": {
        "color": 'black',
        "linestyle": 'dotted',
        "marker": 'o',
        "name": "SortedDims",
    },
    #
    "LowerBound": {
        "color": 'magenta',
        # "linestyle": '-',
        "linestyle": 'solid',
        "marker": '^',
        # "name": "LowerBound(WC)",
        # "name": "LowerBound",
        "name": "WC LowerBound",
    },
    "GT_skln_SIZE": {
        "color": 'magenta',
        "linestyle": 'solid',
        "marker": '^',
        "name": "GT skyline",
    },
}

map__algo__n__feature__distribution = {}
for c_number_of_features in all_features_numbers:
    #
    map__algo__n__feature__distribution, sulist_all_n_values = compute_distributions(input_file_name,
                                                                                     c_number_of_features,
                                                                                     all_feature_names)
    #
    # print()
    # print("map__algo__n__feature__distribution")
    # pp.pprint(map__algo__n__feature__distribution)
    # print()
    # print("sulist_all_n_values")
    # pp.pprint(sulist_all_n_values)
    #
    for alg in map__algo__n__feature__distribution:
        c_distrib = map__algo__n__feature__distribution[alg][5000]['skln_size']
        print()
        print(alg, "5000 points", 'skln_size')
        comp_stat = compute_complete_statistics(c_distrib)
        pp.pprint(comp_stat)
        #
    print()
    #
    for c_feature in all_feature_names:
        #
        plot_title = c_feature + " :)"
        map__algo___x_y_error = {}
        map__algo___x_y = {}
        for c_algo in all_algo_names:
            #
            map__algo___x_y_error[c_algo] = []
            map__algo___x_y[c_algo] = []
            #
            for c_n in sulist_all_n_values:
                c_feature_distribution = map__algo__n__feature__distribution[c_algo][c_n][c_feature]
                map__statistic_indicator__value, sorted_list__statistics_indicator = compute_complete_statistics(
                    c_feature_distribution)
                c_avg = map__statistic_indicator__value["avg"]
                c_std = map__statistic_indicator__value["std"]
                map__algo___x_y_error[c_algo].append((c_n, c_avg, c_std))
                #
                # c_max = map__statistic_indicator__value["max"]
                # map__algo___x_y_error[c_algo].append((c_n, c_max, c_std))
                #
                c_max = map__statistic_indicator__value["max"]
                map__algo___x_y[c_algo].append((c_n, c_max))
        #
        #
        #
        # The strange case of num_comparisons...
        if c_feature == "num_comparisons":
            #
            # The strange case for the Lower-Bound...
            c_algo = all_algo_names[0]
            map__algo___x_y_error["LowerBound"] = []
            map__algo___x_y["LowerBound"] = []
            pathed_c_feature = "max_between_Lower_Bound_with_2_Max_Finding_and_Output_Sensitive_Lower_Bound"
            for c_n in sulist_all_n_values:
                c_feature_distribution = map__algo__n__feature__distribution[c_algo][c_n][pathed_c_feature]
                map__statistic_indicator__value, sorted_list__statistics_indicator = compute_complete_statistics(
                    c_feature_distribution)
                c_avg = map__statistic_indicator__value["avg"]
                c_std = map__statistic_indicator__value["std"]
                map__algo___x_y_error["LowerBound"].append((c_n, c_avg, c_std))
                #
                c_max = map__statistic_indicator__value["max"]
                map__algo___x_y[c_algo].append((c_n, c_max))
            #
        #
        #
        # The strange case of skln_size...
        if c_feature == "skln_size":
            #
            # The strange of skln_size...
            c_algo = all_algo_names[0]
            map__algo___x_y_error["GT_skln_SIZE"] = []
            map__algo___x_y["GT_skln_SIZE"] = []
            pathed_c_feature = "GT_skln_SIZE"
            for c_n in sulist_all_n_values:
                c_feature_distribution = map__algo__n__feature__distribution[c_algo][c_n][pathed_c_feature]
                map__statistic_indicator__value, sorted_list__statistics_indicator = compute_complete_statistics(
                    c_feature_distribution)
                c_avg = map__statistic_indicator__value["avg"]
                c_std = map__statistic_indicator__value["std"]
                # print()
                # print("GT_skln_SIZE")
                # print("c_n", c_n)
                # print("c_avg", c_avg)
                # print("c_std", c_std)
                # print()
                map__algo___x_y_error["GT_skln_SIZE"].append((c_n, c_avg, c_std))
                #
                c_max = map__statistic_indicator__value["max"]
                map__algo___x_y[c_algo].append((c_n, c_max))
            #

        plot_file_name = "./output/" + prefix_output_file_names + "__" + c_feature + "__num_features_" + str(
            c_number_of_features) + "__.pdf"
        plot(plot_title, map__algo___x_y_error, plot_file_name, map__algo__plot_feature__value, c_feature)
        #
        if False and (c_feature == "Accuracy" or c_feature == "Correctness"):
            plot_file_name__MAX = "./output/" + prefix_output_file_names + "__" + c_feature + "__num_features_" + str(
                c_number_of_features) + "__" + "MAX" + "__.pdf"
            plot_2(plot_title, map__algo___x_y, plot_file_name__MAX, map__algo__plot_feature__value, c_feature)
            pp.pprint(map__algo___x_y["ApA"])  ### ofd
