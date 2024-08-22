import numpy as np

# points = [[[1.2,1.2,1.2], [-4.1,5.1,-6.1]],[[5.1,5.2,5.3],[-7.6,8.6,-9.6]]]
points = [[[41.8321425,-83.98233401,13.11693601]], [[43.03235775,-84.5559605,12.14092975]]]
scale = 1

def threed_dda(points, scale):
    def bresenham3D(startPoint, endPoint):
        path = [] 

        startPoint = [int(startPoint[0]),int(startPoint[1]),int(startPoint[2])]
        endPoint = [int(endPoint[0]),int(endPoint[1]),int(endPoint[2])]

        steepXY = (np.abs(endPoint[1] - startPoint[1]) > np.abs(endPoint[0] - startPoint[0]))
        if(steepXY):   
            startPoint[0], startPoint[1] = startPoint[1], startPoint[0]
            endPoint[0], endPoint[1] = endPoint[1], endPoint[0]

        steepXZ = (np.abs(endPoint[2] - startPoint[2]) > np.abs(endPoint[0] - startPoint[0]))
        if(steepXZ):
            startPoint[0], startPoint[2] = startPoint[2], startPoint[0]
            endPoint[0], endPoint[2] = endPoint[2], endPoint[0]

        delta = [np.abs(endPoint[0] - startPoint[0]), np.abs(endPoint[1] - startPoint[1]), np.abs(endPoint[2] - startPoint[2])]

        errorXY = delta[0] / 2
        errorXZ = delta[0] / 2

        step = [
        -1 if startPoint[0] > endPoint[0] else 1,
        -1 if startPoint[1] > endPoint[1] else 1,
        -1 if startPoint[2] > endPoint[2] else 1
        ]

        y = startPoint[1]
        z = startPoint[2]

        for x in range(startPoint[0], endPoint[0], step[0]):
            point = [x, y, z]

            if(steepXZ):
                point[0], point[2] = point[2], point[0]
            if(steepXY):
                point[0], point[1] = point[1], point[0]

            # print(point)
            errorXY -= delta[1]
            errorXZ -= delta[2]

            if(errorXY < 0):
                y += step[1]
                errorXY += delta[0]

            if(errorXZ < 0):
                z += step[2]
                errorXZ += delta[0]

            path.append(point)
        return path
    
    points2 = []
    for i in range(len(points)):
        if i != len(points) - 1:
            for j in range(len(points[i])):
                p1 = [scale * c for c in points[i][j]]
                p2 = [scale * c for c in points[i+1][j]]
                added_points = bresenham3D(p1, p2)
                points2 += added_points
        else:
            for j in points[i]:
                points2 += [[scale * c for c in j]]
    return points2


def threed_dda_2(points, scale):
    def bresenham3D(startPoint, endPoint):
        path = []

        for i in range(3):
            if startPoint[i] < 0:
                startPoint[i] -= 1
            if endPoint[i] < 0:
                endPoint[i] -= 1

        startPoint = [int(startPoint[0]),int(startPoint[1]),int(startPoint[2])]
        endPoint = [int(endPoint[0]),int(endPoint[1]),int(endPoint[2])]

        steepXY = (np.abs(endPoint[1] - startPoint[1]) > np.abs(endPoint[0] - startPoint[0]))
        if(steepXY):   
            startPoint[0], startPoint[1] = startPoint[1], startPoint[0]
            endPoint[0], endPoint[1] = endPoint[1], endPoint[0]

        steepXZ = (np.abs(endPoint[2] - startPoint[2]) > np.abs(endPoint[0] - startPoint[0]))
        if(steepXZ):
            startPoint[0], startPoint[2] = startPoint[2], startPoint[0]
            endPoint[0], endPoint[2] = endPoint[2], endPoint[0]

        delta = [np.abs(endPoint[0] - startPoint[0]), np.abs(endPoint[1] - startPoint[1]), np.abs(endPoint[2] - startPoint[2])]

        errorXY = delta[0] / 2
        errorXZ = delta[0] / 2

        step = [
        -1 if startPoint[0] > endPoint[0] else 1,
        -1 if startPoint[1] > endPoint[1] else 1,
        -1 if startPoint[2] > endPoint[2] else 1
        ]

        y = startPoint[1]
        z = startPoint[2]

        for x in range(startPoint[0], endPoint[0], step[0]):
            point = [x, y, z]

            if(steepXZ):
                point[0], point[2] = point[2], point[0]
            if(steepXY):
                point[0], point[1] = point[1], point[0]

            # print(point)
            errorXY -= delta[1]
            errorXZ -= delta[2]

            if(errorXY < 0):
                y += step[1]
                errorXY += delta[0]

            if(errorXZ < 0):
                z += step[2]
                errorXZ += delta[0]

            path.append(point)
        return path
    
    a_tube_segment = []
    for i in range(len(points)):
        # print(len(points))
        if i != len(points) - 1:
            for j in range(len(points[i])):
                p1 = [scale * c for c in points[i][j]]
                p2 = [scale * c for c in points[i+1][j]]
                a_line_segment = [p1] + bresenham3D(p1[:], p2[:]) + [p2]
                a_tube_segment.append(a_line_segment)
        else:
            for j in range(len(points[i])):
                p1 = [scale * c for c in points[i-1][j]]
                p2 = [scale * c for c in points[i][j]]
                p3 = []
                for c in p2:
                    if c < 0:
                        p3.append(int(c - 1))
                    else:
                        p3.append(int(c))
                a = [p1] + [p3] + [p2]
                a_tube_segment.append(a)
    return a_tube_segment

if __name__ == '__main__':
    print(threed_dda_2(points, scale))