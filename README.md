# susceptibility-to-fieldmap-fft

# Table of contents
1. [Theory](#theory)
2. [Installation](#installation)
3. [Usage](#usage)
4. [References](#references)

## Theory 
The theory section is taken from [Fourier-based-field-estimation](https://github.com/evaalonsoortiz/Fourier-based-field-estimation).

The "susceptibility-to-fieldmap-fft" code allows one to estimate the magnetic field perturbation that arises when an object is placed within a magnetic field.

When an object is placed within an MRI scanner, it is assumed that the magnetic field experienced by the object is uniform and equal to the applied ( $B_0$ ) field. However, magnetic susceptibility ( $\chi$ ) differences between tissues in the body will lead to a non-uniform magnetic field within the body. This inhomogeneous magnetic field will cause image artefacts. These artefacts can be corrected for, if the magnetic field distribution is known. For this, an accurate map of the magnetic field must be acquired. This code allows the user to simulate magnetic fields, which can be useful for validating acquired field maps. 

In MRI, the $B_0$ field is aligned along the z-axis. When an object is placed within this field, it will become magnetized and only the z-component of the induced magnetization will be significant. The Fourier transform of z-component of the induced magnetic field can be expressed as follows (see [Marques et al.](https://onlinelibrary.wiley.com/doi/10.1002/cmr.b.20034) for a full derivation of this expression):

$$ \tilde B_{dz} (\mathbf{k}) = \tilde M_{z} (\mathbf{k}) \cdot \mu_0 \bigg (\frac{1}{3} - \frac{k_z^2}{|\mathbf{k}|^2} \bigg) $$ 

where the spatial frequency, $k$ is equal to $|k|^2=k_x^2+k_y^2+k_z^2$, $\mu_0$ is the permeability of free space, and $M_z$ is the induced magnetization along the z-axis and equal to:

$$ M_{z} (\mathbf{r}) = \chi(\mathbf{r}) \frac{B_0}{\mu_0 (1 + \chi(\mathbf{r}))} $$ 

If $\chi << 1 $, then we can approximate $M_{z} (\mathbf{r})$ as:

$$ M_{z} (\mathbf{r}) \approx \chi(\mathbf{r}) \frac{B_0}{\mu_0} $$

The first equation can then be rewritten as:

$$ \tilde B_{dz} (\mathbf{k}) = \tilde \chi (\mathbf{k}) \cdot B_0 \bigg (\frac{1}{3} - \frac{k_z^2}{|\mathbf{k}|^2} \bigg) $$

This equation allows us to simulate the magnetic field perturbation arising from a susceptibility distribution $\chi(r)$ when introduced within $B_0$. 

It should be noted that when $k=0$, the equation is undefined. $k=0$ is the spatial frequency with wavelength equal to zero, and $\tilde B_{dz} (\mathbf{k = 0})$ is otherwise interpreted as the average field. In order to avoid a singularity, one must assign a value to $\tilde B_{dz} (\mathbf{k} = 0)$, and for this, some assumptions must be made. 

### Setting the value of $\tilde B_{dz} (\mathbf{k} = 0)$ when the average magnetic field does not equal zero

In order to determine the appropriate value to assign to  $\tilde B_{dz} (\mathbf{k} = 0)$ we can consider two scenarios. 

#### Scenario 1: Sphere in an infinite medium

<p align="center">
<img src="https://user-images.githubusercontent.com/112189990/194596500-c4b6450d-8d6e-41f8-a768-fbed345f261e.png" width="200" height="230">
</p>

The derivation for the analytical solution of the magnetic field arising from a sphere placed within an infinite medium is given in Brown et al. This solution includes the Lorentz sphere correction. If the background material has a susceptibility of $\chi_e$ and sphere has a susceptibility of $\chi_i$, the magnetic field inside and outside of the sphere is expressed as:

- Internal field: $\frac{1}{3} \chi_e B_0$
- External field: $\frac{1}{3} (\chi_i - \chi_e) \cdot \frac{a^3}{r^3} (3 \cos^2(\theta) - 1) \cdot B_0 + \frac{1}{3} \chi_e B_0$

From this the average field value can be derived. For $r >> a$ , we can see that both the internal and external field will go to a value of $\frac{1}{3} \chi_e B_0$. $\tilde B_{dz} (\mathbf{k} = 0)$ can be set to $\frac{1}{3} \chi_e B_0$.

#### Scenario 2: Infinitely long cylinder in an infinite medium
<p align="center">
<img src="https://user-images.githubusercontent.com/112189990/194596320-76b668d3-5dbd-42f7-881e-e43b82f3653c.png" width="200" height="230">
</p>

The derivation for the analytical solution of the magnetic field arising from an infinite cylinder placed within an infinite medium is given in Brown et al. This solution includes the Lorentz sphere correction. If the background material has a susceptibility of $\chi_e$ and cylinder has a susceptibility of $\chi_i$, the magnetic field inside and outside of the cylinder is expressed as:

- Internal field: $\frac{1}{6} (\chi_i - \chi_e) \cdot (3\cos^2(\theta) - 1) B_0 + \frac{1}{3} \chi_e B_0$
- External field: $\frac{1}{2} (\chi_i - \chi_e) \cdot \frac{a^2}{r^2} \sin^2(\theta) \cos(2\phi) B_0 + \frac{1}{3} \chi_e B_0$

where $\theta$ is the angle between the direction of the main magnetic field and the central axis of the cylinder.

If $r>>a$, then the external field again goes to a value of $\frac{1}{3} \chi_e B_0$. Based on this, we can assume that  $\tilde B_{dz} (\mathbf{k} = 0) = \frac{1}{3} \chi_e B_0$.

### Setting the value of $\tilde B_{dz} (\mathbf{k} = 0)$ when the average magnetic field is equal to zero (i.e., a "demodulated" field)

Signals arising from an MRI scanner will be "demodulated". A consequence of this is that the average magnetic field within a measured field map is set to zero (here we call this a demodulated field) and any deviation from zero is due to susceptibility differences. 

<p align="center">
<img src="https://user-images.githubusercontent.com/112189990/206759060-6093c10d-b072-41ee-beb1-2eae9d184932.png" width="400" height="200">
</p>

In order to simulate this scenario, we can assume that $\tilde B_{dz} (\mathbf{k} = 0) = 0$. If the susceptibility differences between materials is known, then the demodulated field ( $\tilde B_{dz-demod} (\mathbf{k})$ ) can be computed as follows:

$$ \tilde B_{dz-demod} (\mathbf{k}) =  \Delta \tilde \chi (\mathbf{k}) \cdot B_0 \bigg (\frac{1}{3} - \frac{k_z^2}{|\mathbf{k}|^2} \bigg) $$


These final equations are the ones used in the function **compute_bz**, which calculates the magnetic field offset produced by a susceptibility distribution subject to a uniform external magnetic field $B_0$ (oriented along the z-axis).

## Installation 

- Clone the repository

```
git clone https://github.com/shimming-toolbox/susceptibility-to-fieldmap-fft.git
cd susceptibility-to-fieldmap-fft
```

- Create a virtual environnement 

```
conda create --name <name of your environement> python=3.9
conda activate <name of your environment>
```

- Install the package

```
pip install .
```

You will need to ```conda activate <name of your environment>``` each time you want to use the package.

## Usage
Once the package is installed, the commands can be run directly from the terminal. Here is the description of the two commands available.

### compute_fieldmap

The `compute_fieldmap` command allows computation of a $B_0$ fieldmap based on a susceptibility distribution given as an input.

**Inputs** 
- input_file : path to the susceptibility distribution (NIfTI file)
- output_file : path for the fieldmap (NIfTI file)

**Output** 
The calculated fieldmap at the specified path.

Example:
```
compute_fieldmap "inpath/susceptibility_distribution.nii.gz" "outpath/fieldmap.nii.gz"
```

### analytical_cases

The _analytical_cases_ command allows for comparaison between simulated and analytical results for a spherical and cylindrical phantom. 

**Inputs** 
- -t, geometry type : 'spherical' or 'cylindrical'
- -b, buffer (optional, default=2): Buffer value for zero-padding around the phantom

**Outputs** 
Plots to visualize the results

Example:
```
analytical_cases -t "spherical"
```
The figures generated would be 

![alt text](Figure_1.png)

![alt text](Figure_2.png)

## References

J.P. MARQUES, R. BOWTELL Concepts in Magnetic Resonance Part B (Magnetic Resonance Engineering), Vol. 25B(1) 65-78 (2005)

BROWN, W.B., CHENG, Y-C.N., HAACKE, E.M., THOMPSON, M.R. and VENKATESAN, R., Magnetic resonance imaging : physical principles and sequence design, chapter 25 Magnetic Properties of Tissues : Theory and Measurement. John Wiley & Sons, 2014.
