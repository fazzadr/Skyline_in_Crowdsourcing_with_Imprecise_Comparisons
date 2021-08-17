import csv
import pprint as pp
import random
from Point import Point


def create_housing_dataset():
    input_file_name = "./housing_dataset/housing.csv"

    list__field_index_of_interest = [2, 6, 7, 8]
    list__MAX_value_for_each_field_index_of_interest = [float("-inf")] * len(list__field_index_of_interest)
    # max_number_of_objects = 5000
    # max_number_of_objects = 10000
    max_number_of_objects = 20640

    set__tuple = set()

    input_file = open(input_file_name, 'r', encoding="utf-8")
    input_file_csv_reader = csv.reader(input_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE)
    header = input_file_csv_reader.__next__()
    output_header = []
    for field_index in list__field_index_of_interest:
        output_header.append(header[field_index])
    #
    num_records = 0
    for c_record in input_file_csv_reader:
        num_records += 1
        #
        c_output_record = []
        #
        for field_index in list__field_index_of_interest:
            c_output_record.append(float(c_record[field_index]))
        #
        set__tuple.add(tuple(c_output_record))
        # pp.pprint(c_output_record)
        #
    #
    input_file.close()

    print()
    print("|set__tuple|=", len(set__tuple))
    print("num_records =", num_records)
    print()

    list__all_objects = list(set__tuple)
    # list__all_objects.sort(key=lambda x: (x[0], x[1], x[2], x[3],))
    random.shuffle(list__all_objects)

    list__all_selected_objects_with_id = []
    c_id = 0
    for c_tuple in list__all_objects[:max_number_of_objects]:
        c_id += 1
        list__all_selected_objects_with_id.append([c_id] + list(c_tuple))
        for index in range(len(c_tuple)):
            list__MAX_value_for_each_field_index_of_interest[index] = c_tuple[index] if c_tuple[index] > \
                                                                                        list__MAX_value_for_each_field_index_of_interest[
                                                                                            index] else \
                list__MAX_value_for_each_field_index_of_interest[index]
    #
    print("\n list__MAX_value_for_each_field_index_of_interest")
    pp.pprint(list__MAX_value_for_each_field_index_of_interest)
    #
    # The Less The Better
    for rec in list__all_selected_objects_with_id:
        for index in [1, 2]:
            rec[index] = abs(list__MAX_value_for_each_field_index_of_interest[index - 1] - rec[index])
    #

    outout_file_name = "./housing_dataset/" + str(max_number_of_objects) + "__housing.tsv"
    output_file_handler = open(outout_file_name, "w", 1000000)
    csv_writer_skln = csv.writer(output_file_handler, delimiter='\t', quoting=csv.QUOTE_ALL)
    output_header = ["id"] + output_header
    csv_writer_skln.writerow(output_header)
    csv_writer_skln.writerows(list__all_selected_objects_with_id)
    output_file_handler.close()
    #
    return


def load_housing_dataset(complete_input_file_name):
    input_file_handler = open(complete_input_file_name, "r", 1000000)
    csv_reader_skln = csv.reader(input_file_handler, delimiter='\t', quoting=csv.QUOTE_ALL)
    header = csv_reader_skln.__next__()
    list__points = []
    for record in csv_reader_skln:
        out_record = []
        c_id = int(record[0])
        out_record.append(c_id)
        for c_field in record[1:]:
            out_record.append(float(c_field))
        list__points.append(out_record)
    input_file_handler.close()
    return header, list__points


#
##
###
def create_set_of_first_num_points_housing_points_in_the_selected_features(list__points, num_points,
                                                                           list__indexes_of_features):
    #
    set__object_Point = set()
    #
    for c_point in list__points[:num_points]:
        #
        pp.pprint(c_point)
        #
        c_point_id = c_point[0]
        #
        c_point_features = []
        for c_feature_index in list__indexes_of_features:
            c_point_features.append(c_point[c_feature_index])
        #
        c_point_object = Point(components=c_point_features, ID=str(c_point_id))
        set__object_Point.add(c_point_object)
    #
    return set__object_Point


create_housing_dataset()
