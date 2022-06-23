import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri


class Triangle(object):
    def __init__(self, x_nodes, y_nodes, rho_nodes):
        self.x0 = x_nodes[0]
        self.y0 = y_nodes[0]
        self.x1 = x_nodes[1]
        self.y1 = y_nodes[1]
        self.x2 = x_nodes[2]
        self.y2 = y_nodes[2]
        self.rho_nodes = rho_nodes
        self.set_counter_clockwise()

    @staticmethod
    def cross_product(v1, v2):
        return v1[0] * v2[1] - v2[0] * v1[1]

    def is_counter_clockwise(self):
        x01 = self.x1 - self.x0
        x02 = self.x2 - self.x0
        y01 = self.y1 - self.y0
        y02 = self.y2 - self.y0
        return self.cross_product((x01, y01), (x02, y02)) > 0

    def set_counter_clockwise(self):
        if not self.is_counter_clockwise():
            self.x1, self.x2 = self.x2, self.x1
            self.y1, self.y2 = self.y2, self.y1
            self.rho_nodes[1], self.rho_nodes[2] = self.rho_nodes[2], self.rho_nodes[1]

    def contain_point(self, xt, yt):
        x01 = self.x1 - self.x0
        y01 = self.y1 - self.y0
        x0t = xt - self.x0
        y0t = yt - self.y0
        x12 = self.x2 - self.x1
        y12 = self.y2 - self.y1
        x1t = xt - self.x1
        y1t = yt - self.y1
        x20 = self.x0 - self.x2
        y20 = self.y0 - self.y2
        x2t = xt - self.x2
        y2t = yt - self.y2
        return self.cross_product((x01, y01), (x0t, y0t)) >= 0 and \
               self.cross_product((x12, y12), (x1t, y1t)) >= 0 and \
               self.cross_product((x20, y20), (x2t, y2t)) >= 0

    def get_barycentric_interpolation(self, xt, yt, f0, f1, f2):
        assert self.contain_point(xt, yt), f'Point ({xt}, {yt}) is outside the triangle!'
        w0 = ((yt - self.y1) * (self.x2 - self.x1) - (xt - self.x1) * (self.y2 - self.y1)) / \
             ((self.y0 - self.y1) * (self.x2 - self.x1) - (self.x0 - self.x1) * (self.y2 - self.y1))
        w1 = ((yt - self.y2) * (self.x0 - self.x2) - (xt - self.x2) * (self.y0 - self.y2)) / \
             ((self.y1 - self.y2) * (self.x0 - self.x2) - (self.x1 - self.x2) * (self.y0 - self.y2))
        w2 = 1 - w0 - w1
        return f0 * w0 + f1 * w1 + f2 * w2

    def get_interpolated_rho(self, xt, yt):
        rho_interpolate = self.get_barycentric_interpolation(xt, yt, self.rho_nodes[0], self.rho_nodes[1], self.rho_nodes[2])
        return rho_interpolate


def get_triangles(x_points, y_points, rho_nodes, plot=False):
    x_points = np.array(x_points)
    y_points = np.array(y_points)
    triObj = tri.Triangulation(x_points, y_points)
    triangles = [Triangle(x_points[nodes], y_points[nodes], rho_nodes[nodes]) for nodes in triObj.triangles]
    if plot:
        plt.triplot(triObj)
        plt.show()
    return triangles


if __name__ == '__main__':
    grid_x = np.linspace(0, 4, 5)
    grid_y = np.linspace(0, 4, 5)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    triangles = get_triangles(mesh_x.ravel(), mesh_y.ravel())