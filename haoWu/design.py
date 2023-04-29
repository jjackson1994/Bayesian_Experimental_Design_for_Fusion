from forward_2d_square import compute_cross_points, compute_all_chord_lengths, compute_response_matrix, map_profile_to_2d
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


class Interferometry(object):
    def __init__(self):
        self.r = np.linspace(1.0, 2.0, 201)
        self.z = np.linspace(-0.5, 0.5, 201)
        self.R, self.Z = np.meshgrid(self.r, self.z)
        self.rho_1d = np.linspace(0.0, 1.0, 101)
        self.rho_all = ((self.R - 1.5) ** 2 / 0.16 + self.Z ** 2 / 0.25)
        self.start_angles = None
        self.end_angles = None
        self.num_los = 0
        self.response = None
        self.samples = None
        self.sigma = 0.05   # measurement uncertainty level

    def set_angles(self, start_angles, end_angles):
        self.start_angles = start_angles
        self.end_angles = end_angles
        self.num_los = len(self.start_angles)
        cross_points = compute_cross_points(self.r, self.z, self.start_angles, self.end_angles)
        chord_lengths = compute_all_chord_lengths(self.r, self.z, cross_points)
        exclude_indices = np.where(self.rho_all.flatten() > 1)
        chord_lengths[:, exclude_indices] = 0  # Remove the points outside the plasma boundary
        self.response = compute_response_matrix(chord_lengths, self.rho_all.flatten(), self.rho_1d)

    def calculate_simulated_signals(self, dens_1d):
        """
        Given 1D density profile, calculate simulated signals

        :param dens_1d: 1D density profile
        :return: Expected line-integrated signals along the lines of sight
        """
        lid = np.matmul(self.response, dens_1d)
        return lid

    def visualize(self, dens_1d):
        dens_2d = map_profile_to_2d(self.rho_1d, dens_1d, self.rho_all)
        lid = self.calculate_simulated_signals(dens_1d)
        channel = np.arange(1, self.num_los + 1)

        fig = plt.figure(figsize=(10, 4))
        grid = plt.GridSpec(4, 10, wspace=0.5, hspace=0.5)
        ax1 = plt.subplot(grid[0:3, 0:4])
        ax2 = plt.subplot(grid[0:3, 6:10])

        # Plot 2-D electron density profile and the lines of sight
        cs = ax1.pcolormesh(self.R, self.Z, dens_2d, shading='auto')
        theta = np.linspace(0, np.pi * 2, 100)
        x_circle = 1.5 + 0.5 * np.cos(theta)
        y_circle = 0.5 * np.sin(theta)
        ax1.plot(x_circle, y_circle, 'r')
        for i in range(self.num_los):
            x_start = 1.5 + 0.5 * np.cos(self.start_angles[i])
            x_end = 1.5 + 0.5 * np.cos(self.end_angles[i])
            y_start = 0.5 * np.sin(self.start_angles[i])
            y_end = 0.5 * np.sin(self.end_angles[i])
            ax1.plot([x_start, x_end], [y_start, y_end], 'k')
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

    def draw_samples(self, num_samples=1000):
        """
        Assume the density profile is given by a * (1 - rho^2). Draw samples from the joint distribution of a and data

        :param num_samples: Number of samples
        :return:
        """
        self.samples = np.zeros((num_samples, 1+self.num_los+2))
        # Sample from uniform prior
        self.samples[:, 0] = (1.5 - 0.5) * np.random.rand(num_samples) + 0.5
        for i in range(num_samples):
            a = self.samples[i, 0]
            dens_1d = a * (1 - self.rho_1d**2)
            lid = self.calculate_simulated_signals(dens_1d)
            self.samples[i, 1:1+self.num_los] = lid + self.sigma * np.random.randn(self.num_los)
            # likelihood
            self.samples[i, 1+self.num_los] = multivariate_normal.pdf(self.samples[i, 1:1+self.num_los], mean=lid, cov=self.sigma)
            # marginal likelihood
            sum = 0
            a_samples = (1.5 - 0.5) * np.random.rand(100) + 0.5
            for j in range(100):
                a = a_samples[j]
                dens_1d = a * (1 - self.rho_1d ** 2)
                lid = self.calculate_simulated_signals(dens_1d)
                sum += multivariate_normal.pdf(self.samples[i, 1:1+self.num_los], mean=lid, cov=self.sigma)
            self.samples[i, -1] = sum / 100

    def calculate_utility_function(self):
        l = np.log(self.samples[:, -2] / self.samples[:, -1])
        return np.mean(l)


if __name__ == '__main__':
    interf = Interferometry()
    interf.set_angles([0], [1.25 * np.pi])
    dens_1d = (1 - interf.rho_1d ** 2) * 4.0
    interf.visualize(dens_1d)
    interf.draw_samples()
    print(interf.calculate_utility_function())