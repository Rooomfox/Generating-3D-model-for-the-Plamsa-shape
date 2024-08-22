# Find how many points there in each cube.
def find_location_of_cube(x, y, z, d = 1):
    x_2 = (xmax + x) // d + 1
    y_2 = ((ymax + y) // d) * length_x
    z_2 = ((zmax + z) // d) * length_x * length_y
    index = x_2 + y_2 + z_2
    cubes[index] = cubes[index] + 1