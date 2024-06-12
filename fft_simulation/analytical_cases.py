from fft_simulation import compute_bz
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import rotate
import argparse

class Spherical:
    def __init__(self, matrix, image_res, R, sus_diff):
        self.matrix = matrix
        self.image_res = image_res
        self.R  = R
        self.sus_diff = sus_diff

    def mask(self):
        [x, y, z] = np.meshgrid(np.linspace(-(self.matrix[0]-1)/2, (self.matrix[0]-1)/2, self.matrix[0]),
                                np.linspace(-(self.matrix[1]-1)/2, (self.matrix[1]-1)/2, self.matrix[1]),
                                np.linspace(-(self.matrix[2]-1)/2, (self.matrix[2]-1)/2, self.matrix[2]))

        r = np.sqrt(x**2 + y**2 + z**2)

        return r**2 < self.R**2
    
    def volume(self):
        return np.where(self.mask() == True, self.sus_diff, 0)
    
    def analyticial_sol(self):
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
    def __init__(self, matrix, image_res, R, sus_diff, theta=np.pi/2):
        self.matrix = matrix
        self.image_res = image_res
        self.R  = R
        self.sus_diff = sus_diff
        self.theta = theta # rotation angle about the y-axis

    def mask(self):
        [x, y, z] = np.meshgrid(np.linspace(-(self.matrix[0]-1)/2, (self.matrix[0]-1)/2, self.matrix[0]),
                                np.linspace(-(self.matrix[1]-1)/2, (self.matrix[1]-1)/2, self.matrix[1]),
                                np.linspace(-(self.matrix[2]-1)/2, (self.matrix[2]-1)/2, self.matrix[2]))

        r = x**2 + y**2 

        mask = r <= self.R**2

        # Rotate the cylinder
        return rotate(mask, self.theta*180/np.pi, axes=(0, 2), reshape=False, order=1)
    
    def volume(self):
        return np.where(self.mask() == True, self.sus_diff, 0)
    
# TODO: For now the analytical solution is only correct for theta = 90
    def analytical_sol(self):
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
    

def main(args):
    matrix = np.array([128,128,128])
    image_res = np.array([1,1,1]) # mm
    R = 15 # mm
    sus_diff = 9 # ppm

    if args.type == 'spherical':
        # create the susceptibility geometry
        sphere = Spherical(matrix, image_res, R, sus_diff)
        sus_dist = sphere.volume()
        # compute Bz variation
        calculated_Bz = compute_bz(sus_dist, image_res, args.buffer)
        # analytical solution
        Bz_analytical = sphere.analyticial_sol()

        # plot sections of the chi distribution
        fig, axes = plt.subplots(2, 3, figsize=(10, 5), dpi=120)

        h = axes[0,0].imshow(sus_dist[matrix[0] // 2, :, :], origin='lower')
        axes[0,0].set_title('Slice along y-z plane')
        plt.colorbar(h)

        h = axes[0,1].imshow(sus_dist[:, matrix[0] // 2, :], origin='lower')
        axes[0,1].set_title('Slice along x-z plane')
        plt.colorbar(h)

        h = axes[0,2].imshow(sus_dist[:, :, matrix[0] // 2], origin='lower')
        axes[0,2].set_title('Slice along y-z plane')
        plt.colorbar(h)

        # plot section of the b0 field variation

        vmin = np.min(calculated_Bz)*1.1
        vmax = np.max(calculated_Bz)*1.1

        h = axes[1,0].imshow(calculated_Bz[63, :, :],vmin=vmin, vmax=vmax, origin='lower')
        axes[0,0].set_title('Y-Z plane')
        plt.colorbar(h)

        h = axes[1,1].imshow(calculated_Bz[:, 63, :],vmin=vmin, vmax=vmax, origin='lower')
        axes[0,1].set_title('X-Z plane')
        plt.colorbar(h)

        h = axes[1,2].imshow(calculated_Bz[:, :, 63],vmin=vmin, vmax=vmax, origin='lower')
        axes[0,2].set_title('X-Y plane')
        plt.colorbar(h)
        plt.tight_layout()
        plt.show()


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
        calculated_Bz = compute_bz(sus_dist, image_res, args.buffer)

        Bz_analytical_x, Bz_analytical_y = cylinder.analytical_sol()

        # plot sections of the chi distribution
        fig, axes = plt.subplots(2, 3, figsize=(10, 5), dpi=120)

        h = axes[0,0].imshow(sus_dist[matrix[0] // 2, :, :], origin='lower')
        axes[0,0].set_title('Slice along y-z plane')
        plt.colorbar(h)

        h = axes[0,1].imshow(sus_dist[:, matrix[0] // 2, :], origin='lower')
        axes[0,1].set_title('Slice along x-z plane')
        plt.colorbar(h)

        h = axes[0,2].imshow(sus_dist[:, :, matrix[0] // 2], origin='lower')
        axes[0,2].set_title('Slice along y-z plane')
        plt.colorbar(h)

        # plot section of the b0 field variation

        vmin = np.min(calculated_Bz)*1.1
        vmax = np.max(calculated_Bz)*1.1

        h = axes[1,0].imshow(calculated_Bz[63, :, :],vmin=vmin, vmax=vmax, origin='lower')
        axes[0,0].set_title('Y-Z plane')
        plt.colorbar(h)

        h = axes[1,1].imshow(calculated_Bz[:, 63, :],vmin=vmin, vmax=vmax, origin='lower')
        axes[0,1].set_title('X-Z plane')
        plt.colorbar(h)

        h = axes[1,2].imshow(calculated_Bz[:, :, 63],vmin=vmin, vmax=vmax, origin='lower')
        axes[0,2].set_title('X-Y plane')
        plt.colorbar(h)
        plt.tight_layout()
        plt.show()

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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t",
                        dest="type",
                        type=str,
                        required=True,
                        choices=['spherical', 'cylindrical'],
                        help="Type of geometry for the simulation")
    
    parser.add_argument("-b",
                        dest="buffer",
                        type=int,
                        required=True,
                        help="Buffer value for zero-padding.")
    
    args = parser.parse_args()
    main(args)