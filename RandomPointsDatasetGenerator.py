import random
from Point import Point



#
##
def generate_a_random_point_in_unitary_positive_hypersphere_sector(point_as_list_of_coordinates):
    #
    c_value = 0.
    l_2_norm = 2.
    while l_2_norm > 1.:
        #
        l_2_norm = 0.
        for d in range(len(point_as_list_of_coordinates)):
            c_value = random.random()
            l_2_norm += c_value * c_value
            if l_2_norm > 1.:
                break
            point_as_list_of_coordinates[d] = c_value
    #
    return point_as_list_of_coordinates


#
##
###
def create_set_of_random_points_in_unitary_positive_hypersphere_sector(num_points, num_dimensions):
    #
    set__object_Point = set()
    set__point = set()
    c_point_as_list_of_coordinates = [0.] * num_dimensions
    while len(set__point) < num_points:
        #
        generate_a_random_point_in_unitary_positive_hypersphere_sector(c_point_as_list_of_coordinates)
        #
        set__point.add(tuple(c_point_as_list_of_coordinates))
        #
    #
    #
    point_id = 0
    for c_point_as_tuple_of_coordinates in set__point:
        point_id += 1
        c_point = Point(components=c_point_as_tuple_of_coordinates, ID=str(point_id))
        set__object_Point.add(c_point)
    #
    set__point = None
    return set__object_Point


def create_set_of_grid_points_in_triangle(num_points_per_axis):
    #
    set__object_Point = set()
    set__point = set()
    c_point_as_list_of_coordinates = [0., 0.]
    #
    step = 1. / num_points_per_axis
    for x in range(0, num_points_per_axis + 1, 1):
        c_x = x / num_points_per_axis
        for y in range(0, num_points_per_axis + 1, 1):
            c_y = y / num_points_per_axis
            #
            if c_x + c_y > 1.:
                break
            #
            set__point.add((c_x, c_y))
            #
    #
    point_id = 0
    for c_point_as_tuple_of_coordinates in set__point:
        point_id += 1
        c_point = Point(components=c_point_as_tuple_of_coordinates, ID=str(point_id))
        set__object_Point.add(c_point)
    #
    set__point = None
    return set__object_Point

##
#
##

# import pprint as pp
#
# num_points = 10
# num_dimensions = 5
# c_point_as_list_of_coordinates = [0.] * num_dimensions
# generate_a_random_point_in_unitary_positive_hypersphere_sector(c_point_as_list_of_coordinates)
#
# print(c_point_as_list_of_coordinates)
#
#
#
# set__object_Point = create_set_of_random_points_in_unitary_positive_hypersphere_sector(num_points, num_dimensions)
# print()
# print("set__object_Point")
# for p in set__object_Point:
#     print(p)
# print()
