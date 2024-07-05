from fft_simulation.fft_simulation import compute_bz
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import rotate
import click

class Spherical:
    """
    Represents a spherical object in 3D space.

    Attributes:
    - matrix (ndarray): The dimensions of the matrix representing the object (i.e. [128, 128, 128]).
    - image_res (ndarray): The resolution of the image in mm (i.e. [1, 1, 1] mm).
    - R (int): The radius of the sphere in mm.
    - sus_diff (float): The susceptibility difference of the sphere 
                        (sus_diff = susceptibility_in_the_sphere - susceptibility_outside).

    Methods:
    - mask(): Generates a mask representing the spherical object.
    - volume(): Create a volume of the sphere with the corresponding suceptibility value.
    - analyticial_sol(): Calculates the analytical solution for the magnetic field inside and outside the sphere.
    """

    def __init__(self, matrix, image_res, R, sus_diff):
        self.matrix = matrix
        self.image_res = image_res
        self.R  = R
        self.sus_diff = sus_diff

    def mask(self):
        """
        Generates a mask representing the spherical object.

        Returns:
        - mask (ndarray): A boolean array representing the mask.
        """
        [x, y, z] = np.meshgrid(np.linspace(-(self.matrix[0]-1)/2, (self.matrix[0]-1)/2, self.matrix[0]),
                                np.linspace(-(self.matrix[1]-1)/2, (self.matrix[1]-1)/2, self.matrix[1]),
                                np.linspace(-(self.matrix[2]-1)/2, (self.matrix[2]-1)/2, self.matrix[2]))

        r = np.sqrt(x**2 + y**2 + z**2)

        return r**2 < self.R**2
    
    def volume(self):
        """
        Create a volume of the sphere with the corresponding suceptibility value.

        Returns:
        - volume (ndarray): A 3D array representing the distribution of suceptibility.
        """
        return np.where(self.mask() == True, self.sus_diff, 0)
    
    def analyticial_sol(self):
        """
        Calculates the analytical solution for the magnetic field inside and outside the sphere.

        Returns:
        - Bz_analytical (ndarray): A 3D array representing the analytical solution for the magnetic field.
        """
        mask = self.mask()

        [x, y, z] = np.meshgrid(np.linspace(-(self.matrix[0]-1)/2, (self.matrix[0]-1)/2, self.matrix[0]),
                                np.linspace(-(self.matrix[1]-1)/2, (self.matrix[1]-1)/2, self.matrix[1]),
                                np.linspace(-(self.matrix[2]-1)/2, (self.matrix[2]-1)/2, self.matrix[2]))


        r = np.sqrt(x**2 + y**2 + z**2)

        Bz_out = self.sus_diff/3 * (self.R/r)**3 * (3*z**2/r**2 - 1)
        Bz_in = np.zeros(self.matrix)


        Bz_in = np.where(mask, Bz_in, 0)
        Bz_out = np.where(~mask, Bz_out, 0)

        Bz_analytical = Bz_out + Bz_in

        return Bz_analytical
        

class Cylindrical:
    """
    Represents a cylindrical object in 3D space.

    Parameters:
    - matrix (ndarray): The dimensions of the matrix representing the object (i.e. [128, 128, 128]).
    - image_res (ndarray): The resolution of the image in mm (i.e. [1, 1, 1] mm).
    - R (int): The radius of the cylinder in mm.
    - sus_diff (float): The susceptibility difference of the cylinder 
                        (sus_diff = susceptibility_in_the_cylinder - susceptibility_outside).
    - theta (float, optional): The rotation angle about the y-axis. Default is pi/2.

    Methods:
    - mask(): Generates a mask representing the cylinder.
    - volume(): Create a volume of the sphere with the corresponding suceptibility value.
    - analytical_sol(): Calculates the analytical solution for the magnetic field inside and outside the cylinder.
    """

    def __init__(self, matrix, image_res, R, sus_diff, theta=np.pi/2):
        self.matrix = matrix
        self.image_res = image_res
        self.R  = R
        self.sus_diff = sus_diff
        self.theta = theta # rotation angle about the y-axis

    def mask(self):
        """
        Generates a mask representing the cylinder.

        Returns:
        - mask (ndarray): The mask representing the cylinder.
        """
        [x, y, z] = np.meshgrid(np.linspace(-(self.matrix[0]-1)/2, (self.matrix[0]-1)/2, self.matrix[0]),
                                np.linspace(-(self.matrix[1]-1)/2, (self.matrix[1]-1)/2, self.matrix[1]),
                                np.linspace(-(self.matrix[2]-1)/2, (self.matrix[2]-1)/2, self.matrix[2]))

        r = x**2 + y**2 

        mask = r <= self.R**2

        # Rotate the cylinder
        return rotate(mask, self.theta*180/np.pi, axes=(0, 2), reshape=False, order=1)
    
    def volume(self):
        """
        Create a volume of the sphere with the corresponding suceptibility value.

        Returns:
        - volume (ndarray): A 3D array representing the distribution of suceptibility.
        """
        return np.where(self.mask() == True, self.sus_diff, 0)
    
    def analytical_sol(self):
        """
        Calculates the analytical solution for the magnetic field inside and outside the cylinder.

        Returns:
        - Bz_analytical_x (ndarray): The analytical solution for the magnetic field along the x-axis.
        - Bz_analytical_y (ndarray): The analytical solution for the magnetic field along the y-axis.
        """
        phi_x = 0
        phi_y = np.pi/2

        [x, y, z] = np.meshgrid(np.linspace(-(self.matrix[0]-1)/2, (self.matrix[0]-1)/2, self.matrix[0]),
                                np.linspace(-(self.matrix[1]-1)/2, (self.matrix[1]-1)/2, self.matrix[1]),
                                np.linspace(-(self.matrix[2]-1)/2, (self.matrix[2]-1)/2, self.matrix[2]))

        r = np.sqrt(x**2 + y**2 + z**2)

        # solution along the x-axis: phi = 0
        Bz_out_x = self.sus_diff/2 * (self.R/r)**2 * np.sin(self.theta)**2 * np.cos(2*phi_x)
        Bz_out_x = np.where(~self.mask(), Bz_out_x, 0)

        # solution along the x-axis: phi = 90
        Bz_out_y = self.sus_diff/2 * (self.R/r)**2 * np.sin(self.theta)**2 * np.cos(2*phi_y)
        Bz_out_y = np.where(~self.mask(), Bz_out_y, 0)

        Bz_in = np.zeros(self.matrix) + self.sus_diff/6 * (3*np.cos(self.theta) - 1)
        Bz_in = np.where(self.mask(), Bz_in, 0)

        Bz_analytical_x = Bz_out_x + Bz_in
        Bz_analytical_y = Bz_out_y + Bz_in

        return Bz_analytical_x, Bz_analytical_y
    
@click.command(help="Compare the analytical solution to the simulated solution for a spherical or cylindrical geometry.")
@click.option('-t', '--geometry-type',required=True, 
              type=click.Choice(['spherical', 'cylindrical']), 
              help='Type of geometry for the simulation')
@click.option('-b', '--buffer', default=2, 
              help='Buffer value for zero-padding.')
def main(geometry_type, buffer):
    """
    Main function for running the Fourier-based simulation with analytical cases.

    Parameters:
    - geometry_type (str): The type of geometry to simulate ('spherical' or 'cylindrical').
    - buffer (float): The buffer size for the simulation.

    Returns:
    - None

    This function performs the following steps:
    1. Initializes the necessary variables and parameters.
    2. Creates the susceptibility geometry based on the specified geometry type.
    3. Computes the Bz variation using the computed susceptibility distribution and buffer size.
    4. Plots sections of the susceptibility distribution and Bz field variation.
    5. Plots the analytical solution and simulated results for the Bz field variation.

    Note: The function uses a matrix size of [128, 128, 128], an image resolution of [1, 1, 1] mm, a radius of 15 mm
            and a susceptibility difference of 9 ppm for the spherical and cylindrical geometries.
    """

    matrix = np.array([128,128,128])
    image_res = np.array([1,1,1]) # mm
    R = 15 # mm
    sus_diff = 9 # ppm

    if geometry_type == 'spherical':
        # create the susceptibility geometry
        sphere = Spherical(matrix, image_res, R, sus_diff)
        sus_dist = sphere.volume()
        # compute Bz variation
        calculated_Bz = compute_bz(sus_dist, image_res, buffer)
        # analytical solution
        Bz_analytical = sphere.analyticial_sol()

        # plot sections of the chi distribution
        fig, axes = plt.subplots(2, 3, figsize=(10, 5), dpi=120)

        h = axes[0,0].imshow(sus_dist[matrix[0] // 2, :, :], origin='lower')
        axes[0,0].set_title('Slice along y-z plane')
        plt.colorbar(h, label='Susceptibility [ppm]')

        h = axes[0,1].imshow(sus_dist[:, matrix[0] // 2, :], origin='lower')
        axes[0,1].set_title('Slice along x-z plane')
        plt.colorbar(h, label='Susceptibility [ppm]')

        h = axes[0,2].imshow(sus_dist[:, :, matrix[0] // 2], origin='lower')
        axes[0,2].set_title('Slice along y-z plane')
        plt.colorbar(h, label='Susceptibility [ppm]')

        # plot section of the b0 field variation

        vmin = np.min(calculated_Bz)*1.1
        vmax = np.max(calculated_Bz)*1.1

        h = axes[1,0].imshow(calculated_Bz[63, :, :],vmin=vmin, vmax=vmax, origin='lower')
        axes[0,0].set_title('Y-Z plane')
        plt.colorbar(h, label='Field variation [ppm]')

        h = axes[1,1].imshow(calculated_Bz[:, 63, :],vmin=vmin, vmax=vmax, origin='lower')
        axes[0,1].set_title('X-Z plane')
        plt.colorbar(h, label='Field variation [ppm]')

        h = axes[1,2].imshow(calculated_Bz[:, :, 63],vmin=vmin, vmax=vmax, origin='lower')
        axes[0,2].set_title('X-Y plane')
        plt.colorbar(h, label='Field variation [ppm]')
        plt.tight_layout()
        plt.show()

        # comparaison with the analytical solution

        fig, axes = plt.subplots(1, 3, figsize=(10, 3), dpi=120)

        axes[0].plot(np.linspace(-64,64, 128), Bz_analytical[:,63,63], label='Theory')
        axes[0].plot(np.linspace(-64,64, 128), calculated_Bz[:,63,63],'--', label='Simulated')
        axes[0].set_ylim(vmin, vmax)
        axes[0].set_xlabel('x position [mm]')
        axes[0].set_ylabel('Field variation [ppm]')
        axes[0].legend()

        axes[1].plot(np.linspace(-64,64, 128), Bz_analytical[63,:,63], label='Theory')
        axes[1].plot(np.linspace(-64,64, 128), calculated_Bz[63,:,63],'--', label='Simulated')
        axes[1].set_ylim(vmin, vmax)
        axes[1].set_xlabel('y position [mm]')
        axes[1].set_ylabel('Field variation [ppm]')
        axes[1].legend()

        axes[2].plot(np.linspace(-64,64, 128), Bz_analytical[63,63,:], label='Theory')
        axes[2].plot(np.linspace(-64,64, 128), calculated_Bz[63,63,:],'--', label='Simulated')
        axes[2].set_ylim(vmin, vmax)
        axes[2].set_xlabel('z position [mm]')
        axes[2].set_ylabel('Field variation [ppm]')
        axes[2].legend()

        plt.tight_layout()
        plt.show()

    else: # type is "cylindrical"

        # create the susceptibility geometry
        cylinder = Cylindrical(matrix, image_res, R, sus_diff)
        sus_dist = cylinder.volume()

        # compute Bz variation
        calculated_Bz = compute_bz(sus_dist, image_res, buffer)

        Bz_analytical_x, Bz_analytical_y = cylinder.analytical_sol()

        # plot sections of the chi distribution
        fig, axes = plt.subplots(2, 3, figsize=(10, 5), dpi=120)

        h = axes[0,0].imshow(sus_dist[matrix[0] // 2, :, :], origin='lower')
        axes[0,0].set_title('Slice along y-z plane')
        plt.colorbar(h, label='Susceptibility [ppm]')

        h = axes[0,1].imshow(sus_dist[:, matrix[0] // 2, :], origin='lower')
        axes[0,1].set_title('Slice along x-z plane')
        plt.colorbar(h, label='Susceptibility [ppm]')

        h = axes[0,2].imshow(sus_dist[:, :, matrix[0] // 2], origin='lower')
        axes[0,2].set_title('Slice along y-z plane')
        plt.colorbar(h, label='Susceptibility [ppm]')

        # plot section of the b0 field variation

        vmin = np.min(calculated_Bz)*1.1
        vmax = np.max(calculated_Bz)*1.1

        h = axes[1,0].imshow(calculated_Bz[63, :, :],vmin=vmin, vmax=vmax, origin='lower')
        axes[0,0].set_title('Y-Z plane')
        plt.colorbar(h, label='Field variation [ppm]')

        h = axes[1,1].imshow(calculated_Bz[:, 63, :],vmin=vmin, vmax=vmax, origin='lower')
        axes[0,1].set_title('X-Z plane')
        plt.colorbar(h, label='Field variation [ppm]')

        h = axes[1,2].imshow(calculated_Bz[:, :, 63],vmin=vmin, vmax=vmax, origin='lower')
        axes[0,2].set_title('X-Y plane')
        plt.colorbar(h, label='Field variation [ppm]')
        plt.tight_layout()
        plt.show()

        # comparaison with the analytical solution

        fig, axes = plt.subplots(1, 3, figsize=(10, 3), dpi=120)

        axes[0].plot(np.linspace(-64,64, 128), Bz_analytical_x[:,63,63], label='Theory')
        axes[0].plot(np.linspace(-64,64, 128), calculated_Bz[:,63,63],'--', label='Simulated')
        axes[0].set_ylim(vmin, vmax)
        axes[0].set_xlabel('x position [mm]')
        axes[0].set_ylabel('Field variation [ppm]')
        axes[0].legend()

        axes[1].plot(np.linspace(-64,64, 128), Bz_analytical_y[63,:,63], label='Theory')
        axes[1].plot(np.linspace(-64,64, 128), calculated_Bz[63,:,63],'--', label='Simulated')
        axes[1].set_ylim(vmin, vmax)
        axes[1].set_xlabel('y position [mm]')
        axes[1].set_ylabel('Field variation [ppm]')
        axes[1].legend()

        axes[2].plot(np.linspace(-64,64, 128), Bz_analytical_x[63,63,:], label='Theory')
        axes[2].plot(np.linspace(-64,64, 128), calculated_Bz[63,63,:],'--', label='Simulated')
        axes[2].set_ylim(vmin, vmax)
        axes[2].set_xlabel('z position [mm]')
        axes[2].set_ylabel('Field variation [ppm]')
        axes[2].legend()

        plt.tight_layout()
        plt.show()
