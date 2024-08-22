import numpy as np

def calculate_vertices_values(xmax, ymax, zmax, points, cubes = {}, vertices = {}):
	"""calculate the distance between vertices of each cube and the line segment."""
	def find_location_of_cube_vertices(index, point):
		x_2, y_2, z_2 = point[0], point[1], point[2]
		if index in cubes.keys():
			return cubes[index]
		else:
			v_1 = (x_2, y_2, z_2)
			v_2 = (x_2 + 1, y_2, z_2)
			v_3 = (x_2 + 1, y_2 + 1, z_2)
			v_4 = (x_2, y_2 + 1, z_2)
			v_5 = (x_2, y_2, z_2 + 1)
			v_6 = (x_2 + 1, y_2, z_2 + 1)
			v_7 = (x_2 + 1, y_2 + 1, z_2 + 1)
			v_8 = (x_2, y_2 + 1, z_2 + 1)
			cubes[index] = [v_1, v_2, v_3, v_4, v_5, v_6, v_7, v_8]
			return cubes[index]

	def calculate_distance(p, q, v_s):
		p = np.array(p)
		q = np.array(q)
		l_s = p - q
		distance_list = []
		for vertex in v_s:
			vertex = np.array(vertex)
			t = np.dot(vertex - q, l_s) / np.dot(l_s, l_s)
			if 0 <= t <= 1:
				distance = np.linalg.norm(t * (p - q) + q - vertex)
			else:
				d1 = np.linalg.norm(vertex - p)
				d2 = np.linalg.norm(vertex - q)
				# if d1 > 1.733 and d2 > 1.733:
				# 	print(p, q, vertex)
				# 	break
				# else:
				if d1 > d2:
					distance = d2
				else:
					distance = d1
			distance_list.append(distance)
		return distance_list

	for point in points[1:-1]:
		num_x = point[0] + xmax
		num_y = point[1] + ymax
		num_z = point[2] + zmax
		length_x = 2 * xmax
		length_y = 2 * ymax
		length_z = 2 * zmax
		index = num_x + num_y * length_x + num_z * length_x * length_y
		index = int(index)
		d_l = calculate_distance(points[0], points[-1], find_location_of_cube_vertices(index, point))
		if index in vertices.keys():
			for i in range(len(vertices[index])):
				a = vertices[index][i]
				b = d_l[i]
				if a > b:
					vertices[index][i] = b
		else:
			vertices[index] = d_l
	# print(vertices[0][1])
	# print(cubes)
	# return vertices

if __name__ == '__main__':
	cubes = {}
	vertices = {}
	xmax, ymax, zmax = 120, 120, 120
	length_x, length_y, length_z = 240, 240, 240
	points = [[-0.5, -0.5, -0.5],[0, 0, 0],[1, 1, 1],[0.5, 0.5, 0.5]]
	# points_2 = [[-0.9, -0.9, -0.9], [-1, -1, -1],[0, 0, 0],[1.9, 1.9, 1.9]]
	points_2 = [[41.8321425, -83.98233401, 13.11693601], [41, -84, 13], [42, -84, 13], [43.03235775, -84.5559605, 12.14092975]]

	# calculate_vertices_values(xmax, ymax, zmax, points, cubes, vertices)
	calculate_vertices_values(xmax, ymax, zmax, points_2, cubes, vertices)
	print(vertices)
	# print(vertices[7][6])