import numpy as np
import math

# filename = 'data/FFHR-d1A_R3.50_20181105_extract.dat'

# with open(filename) as file_object:
# 	lines = file_object.readlines()

# Newlist = []
# for line in lines:
# 	Newlist.append(line.split())

# points = []
# counter = 0
# for line in range(len(Newlist)):
# 		if Newlist[line] != []:
# 			if counter == 90:
# 				n = int(Newlist[line][0])
# 				t_a = float(Newlist[line][3])
# 				x = float(Newlist[line][4])
# 				y = float(Newlist[line][5])
# 				z = float(Newlist[line][6])
# 				br = float(Newlist[line][8])
# 				bz = float(Newlist[line][9])
# 				bt = float(Newlist[line][10])
# 				b = float(Newlist[line][11])
# 				point = (n, t_a, x, y, z, br, bz, bt, b)
# 				points.append(point)
# 		else:
# 			counter += 1
# print(f'The line index is {points[0][0]}')

def larmor_radius(points):
	def calculate_larmor_radius(T, m, q, b):
		"""Calculate the Larmor radius"""
		Ek = T * 1.602e-19
		r = (2 * Ek * m) ** 0.5 / (q * b)
		return(r)

	def calculate_normal_vector(br, bz, bt, t_a):
		"""Transition between the two coordinates"""
		C = bz
		pi = math.pi
		t_a = math.radians(t_a)
		if 0 <= t_a < 0.5 * pi:
			A = br * math.cos(t_a) - bt * math.sin(t_a)
			B = br * math.sin(t_a) + bt * math.cos(t_a)
		elif 0.5 * pi <= t_a < pi:
			A = br * math.cos(pi - t_a) - bt * math.sin(pi - t_a)
			B = br * math.sin(pi - t_a) - bt * math.cos(pi - t_a)
		elif pi <= t_a < 1.5 * pi:
			A = br * math.cos(t_a - pi) + bt * math.sin(t_a - pi)
			B = br * math.sin(t_a - pi) - bt * math.cos(t_a - pi)
		else:
			A = br * math.cos(2 * pi - t_a) + bt * math.sin(2 * pi - t_a)
			B = br * math.sin(2 * pi - t_a) + bt * math.cos(2 * pi - t_a)
		return((A, B, C))

	def calculate_start_point(n_v_1, n_v_2, x, y, z, r):
		"""Using cross product to calculate the start point"""
		A2 = n_v_1[1] * n_v_2[2] - n_v_1[2] * n_v_2[1]
		B2 = n_v_1[2] * n_v_2[0] - n_v_1[0] * n_v_2[2]
		C2 = n_v_1[0] * n_v_2[1] - n_v_1[1] * n_v_2[0]
		try:
			s_x = r ** 2 / (1 + (B2 / A2) ** 2 + (C2 / A2) ** 2)
		except ZeroDivisionError:
			print('we got a zero division error')
			print(f'A2 = {A2}')
		p_x = s_x ** 0.5
		n_x = -p_x
		p_y = p_x * B2 / A2
		n_y = -p_y
		p_z = p_x * C2 / A2
		n_z = -p_z
		p1 = (p_x + x, p_y + y, p_z + z)
		p2 = (n_x + x, n_y + y, n_z + z)
		return(A2, B2, C2, p1, p2)

	def rotation(ori_v, ori_k, theta):
		"""Using Rodrigues' rotation formula"""
		other_points = []
		num = int(360 / theta)
		for i in range(num - 1):
			theta2 = math.radians(int(theta * (i + 1)))
			v = np.array([ori_v[0], ori_v[1], ori_v[2]])
			k = [ori_k[0], ori_k[1], ori_k[2]]
			length = (sum(value**2 for value in k))**0.5
			k_norm = np.array([1 / length * value for value in k])

			c_t = np.cos(theta2)
			s_t = np.sin(theta2)
			cross = np.cross(k_norm, v)
			dot = np.dot(k_norm, v)
			v_rot = list(v * c_t + cross * s_t + k_norm * dot * (1 - c_t))
			other_points.append(v_rot)
		return other_points

	T = 3.5e6
	m = 4 * 1.673e-27
	q = 2 * 1.602e-19
	rota_theta = 45
	origin_points = []
	array = []
	A21, B21, C21 = 0, 0, 0
	label = 1

	for i in range(len(points)):
		if i != len(points) - 1:
			br, bz, bt = points[i][5:8]
			t_a = points[i][1]
			br_2, bz_2, bt_2 = points[i+1][5:8]
			t_a_2 = points[i+1][1]
		else:
			br, bz, bt = points[i-1][5:8]
			t_a = points[i-1][1]
			br_2, bz_2, bt_2 = points[i][5:8]
			t_a_2 = points[i][1]

		n_v_1 = calculate_normal_vector(br, bz, bt, t_a)
		n_v_2 = calculate_normal_vector(br_2, bz_2, bt_2, t_a_2)
		x, y, z = points[i][2:5]
		p0 = [x, y, z]
		r = calculate_larmor_radius(T, m, q, points[i][8])
		A2, B2, C2, p1, p2 = calculate_start_point(n_v_1, n_v_2, x, y, z, r)
		pp1 = [p1[j] - p0[j] for j in range(len(p1))]
		pp2 = [p2[j] - p0[j] for j in range(len(p2))]
		origin_points.append((x, y, z))

		if abs(A21 + A2) < abs(A21) + abs(A2) or abs(B21 + B2) < abs(B21) + abs(B2) and abs(C21 + C2) < abs(C21) + abs(C2):
			label = -label
		if label == 1:
			# array.append([p1] + [p2])
			other_points = rotation(pp1, n_v_1, rota_theta)
			for point in other_points:
				point[0] += x
				point[1] += y
				point[2] += z
			array.append([p1] + other_points)
		else:
			# array.append([p2] + [p1])
			other_points = rotation(pp2, n_v_1, rota_theta)
			for point in other_points:
				point[0] += x
				point[1] += y
				point[2] += z
			array.append([p2] + other_points)
		A21, B21, C21 = A2, B2, C2
	return array