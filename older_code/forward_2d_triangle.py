import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from triangle import get_triangles
from scipy.interpolate import interp1d
from scipy.spatial import distance


mpl.rcParams["image.cmap"] = 'jet'
mpl.rcParams["lines.linewidth"] = 0.5
mpl.rcParams["lines.markersize"] = 4
mpl.rcParams["figure.dpi"] = 200


def get_nearest_normalized_radii(rho_pixel, rho_discrete):
    '''
    Get the nearest discrete normalized radii

    :param rho_pixel: Normalized radii of all pixels
    :param rho_discrete: Discrete normalized radii
    :return: The indices of the discrete values to which the normalized radius of each point belongs.
    '''
    rho_discrete = np.expand_dims(rho_discrete, -1)
    rho_pixel = np.expand_dims(rho_pixel, -1)
    distances = distance.cdist(rho_discrete, rho_pixel)  # Distance between each pair of the two collections of inputs
    digitized = np.argmin(distances, axis=0)
    return digitized


class LineOfSight(object):
    def __init__(self, theta1, theta2, triangles_all, rho_discrete):
        self.center = np.array([1.5, 0.0])  # Center of the circle
        self.r = 0.5    # Radius of the circle
        self.start_point = np.array([self.center[0] + self.r * np.cos(theta1),
                                     self.center[1] + self.r * np.sin(theta1)])
        self.end_point = np.array([self.center[0] + self.r * np.cos(theta2),
                                   self.center[1] + self.r * np.sin(theta2)])
        self.dl = 0.01  # Step length of the integral (m)
        self.num_points = 0  # Number of points discretizing the chord (all points should be inside the plasma)
        self.points = []  # Coordinates of the discrete points on LOS
        self.triangles_all = triangles_all   # All triangles inside the plasma
        self.triangles_of_points = []   # The corresponding triangle to which the point belongs to
        self.rhos = None    # Normalized radii of the discrete points on LOS
        self.rho_discrete = rho_discrete    # Discrete normalized radii used for density reconstruction
        self.chord_length = np.zeros_like(rho_discrete)

        print('generating_points...')
        self.generate_points()
        print('computing normalized radii...')
        self.compute_chord_length()

    def generate_points(self):
        '''
        Generate discrete points on LOS
        '''
        vec = self.end_point - self.start_point
        unit = vec / np.linalg.norm(vec)
        p = self.start_point.copy()
        triangle = self.find_triangle(p[0], p[1])
        # Find the first point inside the plasma
        while triangle is None and \
                np.abs(p[0] - self.start_point[0]) < np.abs(self.end_point[0] - np.abs(self.start_point[0])):
            p += self.dl * unit
            triangle = self.find_triangle(p[0], p[1])
        # Find all points inside the plasma
        while triangle is not None:
            self.points.append(p.copy())
            self.triangles_of_points.append(triangle)
            p += self.dl * unit
            triangle = self.find_triangle(p[0], p[1])
        self.num_points = len(self.points)

    def find_triangle(self, x, y):
        '''
        Return the triangle that contains the given point (x, y)
        '''
        for triangle in self.triangles_all:
            if triangle.contain_point(x, y):
                return triangle
        return None

    def compute_chord_length(self):
        '''
        Compute the contribution to the density profile (at each normalized radius)
        '''
        self.rhos = np.zeros(self.num_points)
        if self.num_points > 0:
            for i in range(self.num_points):
                self.rhos[i] = self.triangles_of_points[i].get_interpolated_rho(self.points[i][0], self.points[i][1])
            if self.rho_discrete is not None:
                digitized = get_nearest_normalized_radii(self.rhos, self.rho_discrete)
                for i in range(self.chord_length.shape[0]):
                    self.chord_length[i] = self.dl * (digitized == i).sum()


def compute_response_matrix(start_angles, end_angles, rho_discrete, triangles_all):
    '''
    Given geometry of lines of sight, compute the full response matrix (mapping from 1-D density profile to line-integrated density)
    :param start_angles: angles that parameterize the start points of lines of sight
    :param end_angles: angles that parameterize the end points of lines of sight
    :param rho_discrete: coordinates of the density profile
    :param triangles_all: Triangle instances to discretize the cross section
    :return:
    '''
    chords = []
    num_los = len(start_angles)
    len_rho_discrete = len(rho_discrete)
    for i in range(num_los):
        print(f'Computing chord {i+1}...')
        chords.append(LineOfSight(theta1=start_angles[i],
                                  theta2=end_angles[i],
                                  triangles_all=triangles_all, rho_discrete=rho_discrete))
    # rho_all = np.hstack([chord.rhos for chord in chords])
    # plt.hist(rho_all)
    # plt.scatter(rho_all, np.ones_like(rho_all))
    plt.show()
    response = np.zeros([num_los, len_rho_discrete])
    for i in range(num_los):
        response[i, :] = chords[i].chord_length
    return response


def map_profile_to_2d(rho_1d, dens_1d, rho_2d):
    '''
    Map 1-D electron density profile to 2-D
    '''
    rho_2d_flat = rho_2d.flatten()
    dens_2d_flat = np.zeros_like(rho_2d_flat)
    indices_inside = np.argwhere(rho_2d_flat <= 1.0).flatten()
    dens_2d_flat[indices_inside] = interp1d(rho_1d, dens_1d, kind='cubic')(rho_2d_flat[indices_inside])
    dens_2d_flat[rho_2d_flat > 1.0] = np.nan
    return dens_2d_flat.reshape(rho_2d.shape)


def plot_example_configuration(start_angles, end_angles, rho_1d, dens_1d):
    r = np.linspace(1.0, 2.0, 101)
    z = np.linspace(-0.5, 0.5, 101)
    R, Z = np.meshgrid(r, z)
    rho_all = ((R - 1.5) ** 2 / 0.16 + Z ** 2 / 0.25)
    dens_2d = map_profile_to_2d(rho_1d, dens_1d, rho_all)
    num_los = len(start_angles)

    r_valid = R[rho_all <= 1]
    z_valid = Z[rho_all <= 1]
    rho_valid = rho_all[rho_all <= 1]
    triangles_all = get_triangles(r_valid, z_valid, rho_valid, plot=False)
    response = compute_response_matrix(start_angles, end_angles, rho_1d, triangles_all)
    lid = np.matmul(response, dens_1d)
    print(lid)
    channel = np.arange(1, num_los + 1)

    fig = plt.figure(figsize=(10, 4))
    grid = plt.GridSpec(4, 10, wspace=0.5, hspace=0.5)
    ax1 = plt.subplot(grid[0:3, 0:4])
    ax2 = plt.subplot(grid[0:3, 6:10])

    # Plot 2-D electron density profile and the lines of sight
    cs = ax1.pcolormesh(R, Z, dens_2d, shading='auto')
    theta = np.linspace(0, np.pi*2, 100)
    x_circle = 1.5 + 0.5 * np.cos(theta)
    y_circle = 0.5 * np.sin(theta)
    ax1.plot(x_circle, y_circle, 'r')
    los_r_start = 1.5 + 0.5 * np.cos(start_angles)
    los_r_end = 1.5 + 0.5 * np.cos(end_angles)
    los_z_start = 0.5 * np.sin(start_angles)
    los_z_end = 0.5 * np.sin(end_angles)
    for i in range(num_los):
        ax1.plot([los_r_start[i], los_r_end[i]], [los_z_start[i], los_z_end[i]], 'k')
    ax1.set_aspect('equal')
    ax1.set_xlabel('R (m)')
    ax1.set_ylabel('Z (m)')
    ax1.set_title('2-D distribution of electron density')
    # cb = fig.colorbar(cs, ax=ax1)

    # Plot the line-integrated electron density for each LOS
    ax2.scatter(channel, lid)
    ax2.plot(channel, lid)
    ax2.set_xlabel('Channel number')
    ax2.set_xticks(channel)
    ax2.set_ylabel('Line integrated density')
    ax2.set_title('Calculated signals')
    plt.show()


if __name__ == '__main__':
    rho_1d = np.linspace(0.0, 1.0, 101)
    dens_1d = (1 - rho_1d**2) * 4.0
    plot_example_configuration([0, 0, 0.5 * np.pi], [np.pi, 1.25 * np.pi, 1.5 * np.pi], rho_1d, dens_1d)
