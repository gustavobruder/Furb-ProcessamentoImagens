class Gallery:
    def __init__(self, size, points):
        points.append(points[0])
        points.append(points[1])
        self.size = size
        self.points = points


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def calculate_counterclockwise(point_o, point_a, point_b):
    return (point_a.x - point_o.x) * (point_b.y - point_o.y) - (point_a.y - point_o.y) * (point_b.x - point_o.x)


def validate_convex(gallery):
    first_ccw = calculate_counterclockwise(gallery.points[0], gallery.points[1], gallery.points[2])

    for index in range(gallery.size):
        point1 = gallery.points[index]
        point2 = gallery.points[index + 1]
        point3 = gallery.points[index + 2]

        ccw = calculate_counterclockwise(point1, point2, point3)

        if first_ccw >= 0 and ccw < 0:
            return False
        elif first_ccw < 0 and ccw > 0:
            return False

    return True


def read_input_data():
    galleries = []

    while True:
        gallery_size = int(input())

        if gallery_size == 0:
            break

        points = []
        for index in range(gallery_size):
            point_coordinates = input().split()
            points.append(Point(int(point_coordinates[0]), int(point_coordinates[1])))

        galleries.append(Gallery(gallery_size, points))

    return galleries


if __name__ == '__main__':
    galleries = read_input_data()

    for gallery in galleries:
        is_convex = validate_convex(gallery)
        print('No' if is_convex else 'Yes')
