import numpy as np
import nibabel as nib
import click
from scipy import fft as sfft
from time import perf_counter
from pathlib import Path

def is_nifti(filepath):
    """
    Check if the given filepath represents a NIfTI file.

    Args:
        filepath (str): The path of the file to check.

    Returns:
        bool: True if the file is a NIfTI file, False otherwise.
    """
    if filepath.endswith('.nii') or filepath.endswith('.nii.gz'):
        return True
    else:
        return False

def load_sus_dist(filepath):
    """
    Load the susceptibility distribution from a given file.

    Args:
        filepath (str): The path to the file containing the susceptibility distribution.

    Returns:
        tuple: A tuple containing the loaded susceptibility distribution as a numpy array 
            and the image resolution as a numpy array.
    """

    image = nib.load(filepath)
    susceptibility_distribution = image.get_fdata()
    header = image.header 
    image_resolution = np.array(header.get_zooms())
    affine_matrix = image.affine

    return susceptibility_distribution, image_resolution, affine_matrix


def compute_bz_reference(susceptibility_distribution, image_resolution=np.array([1,1,1]), buffer=50, mode='edge'):
    """
    Reference implementation, kept verbatim as the numerical oracle.

    This is the pre-refactor solver as of commit 5abb079. It builds the full
    complex spectrum and a full-size kernel via meshgrid, so it is far too
    memory-hungry for production volumes - but it is what minted the golden
    fixtures in tests/fixtures/, and tests/functions/test_refactor_invariance.py
    holds compute_bz to it.

    Compute the Bz field variation in ppm based on a susceptibility distribution
    using a Fourier-based method.

    Args:
        susceptibility_distribution (numpy.ndarray): The 3D array representing the susceptibility distribution.

        image_resolution (numpy.ndarray, optional): The resolution of the image in each dimension. Defaults to [1, 1, 1].

        buffer (int, optional): The buffer size (voxels) for the k-space grid on each size. Defaults to 50.

        mode (str, optional): The padding mode for the susceptibility distribution. Defaults to 'edge'.

    Returns:
        volume_without_buffer (numpy.ndarray): The computed magnetic field Bz in ppm.

    """

    # Pad the susceptibility distribution

    if mode == 'b0SimISMRM':
            susceptibility_distribution=np.pad(susceptibility_distribution, ((buffer, buffer), (0,buffer), (buffer,buffer)), mode='edge')
            susceptibility_distribution=np.pad(susceptibility_distribution, ((0,0),(buffer,0),(0,0)), mode='constant', constant_values=0.35)
    else:
        susceptibility_distribution=np.pad(susceptibility_distribution, buffer, mode=mode)

    # dimensions needs to be a numpy.array
    dimensions = np.array(susceptibility_distribution.shape)

    kmax = 1/(2*image_resolution)

    interval = 2 * kmax / dimensions


    kx_min_shift = (dimensions[0]%2)*interval[0]/2
    ky_min_shift = (dimensions[1]%2)*interval[1]/2
    kz_min_shift = (dimensions[2]%2)*interval[2]/2

    kx_max_shift = -interval[0] + (dimensions[0]%2)*interval[0]/2
    ky_max_shift = -interval[1] + (dimensions[1]%2)*interval[1]/2
    kz_max_shift = -interval[2] + (dimensions[2]%2)*interval[2]/2 


    [kx, ky, kz] = np.meshgrid(np.linspace(-kmax[0] + kx_min_shift, kmax[0] + kx_max_shift, dimensions[0]),
                                np.linspace(-kmax[1] + ky_min_shift, kmax[1] + ky_max_shift, dimensions[1]),
                                np.linspace(-kmax[2] + kz_min_shift, kmax[2] + kz_max_shift, dimensions[2]), indexing='ij')

    # FFT procedure
    # undetermined at the center of k-space
    k2 = kx**2 + ky**2 + kz**2

    with np.errstate(divide='ignore', invalid='ignore'):
        x_kernel = 1/3 - kz**2/k2

        x_kernel[int(dimensions[0]/2-1/2*(dimensions[0]%2)), int(dimensions[1]/2-1/2*(dimensions[1]%2)), int(dimensions[2]/2-1/2*(dimensions[2]%2))] = 1/3
        
        kernel = np.fft.ifftshift(x_kernel)

    FFT_chi = np.fft.fftn(susceptibility_distribution, dimensions)

    Bz_fft = kernel*FFT_chi

    # retrive the inital FOV

    volume_with_buffer = np.real(np.fft.ifftn(Bz_fft))

    if buffer == 0:
        volume_without_buffer = volume_with_buffer
    else:
        volume_without_buffer = volume_with_buffer[buffer:-buffer, buffer:-buffer, buffer:-buffer]

    return volume_without_buffer


def fast_padded_shape(n, zerofill, real_axis=False):
    """Padded length for one axis: at least ``zerofill * n``, rounded up to a
    length the FFT can factor efficiently.

    An FFT gets its speed by recursively splitting a transform of length N into
    smaller ones using N's prime factors. Small factors (2, 3, 5, 7) split
    cleanly; a large prime factor cannot be split at all, so the library falls
    back to Bluestein's algorithm, which re-expresses that transform as a
    convolution and internally pads to a friendlier length. It still works, it
    just costs several times more.

    Exact doubling walks straight into this: sub-amuPA's canvas is 547 voxels
    across, and 2 x 547 = 1094, whose only factors are 2 and the prime 547.
    Measured on a (1094, 300, 320) volume against (1100, 300, 320) - which is
    2^2 x 5^2 x 11, and 0.6% LARGER:

        rfftn, whole volume :  0.10 s  vs  0.05 s   (2x slower)
        rfft, that axis only:  2.9 ms  vs  0.5 ms   (6x slower)

    For reference a fully prime 1093 costs 3.2 ms, about the same as 1094, which
    confirms the 547 factor is what does the damage; while 1024 = 2^10 costs
    0.8 ms, so a pure power of two is not the target either - a mixture of small
    factors wins.

    Rounding up costs ~2.5% more voxels across this dataset and only ever ADDS
    padding, so the ratio the caller asked for remains a lower bound.

    Args:
        n (int): original axis length.
        zerofill (float): requested ratio, e.g. 2 for double.
        real_axis (bool): True for the axis rfftn transforms as real (the last
            one), which has a different set of efficient lengths.

    Returns:
        int: the padded length.
    """
    target = int(np.ceil(zerofill * n))
    return int(sfft.next_fast_len(target, real=real_axis))


def zerofill_padded_shape(shape, zerofill):
    """Per-axis padded shape ``compute_bz`` will use for a given zerofill ratio.

    Exposed separately so the run driver can predict peak memory per subject
    before committing a machine to it, and so tests can check that the last axis
    - the one ``rfftn`` transforms as real - uses the real-FFT length table.

    Args:
        shape (tuple): original volume shape.
        zerofill (float): requested ratio.

    Returns:
        tuple: padded shape, one entry per axis.
    """
    last = len(shape) - 1
    return tuple(fast_padded_shape(n, zerofill, real_axis=(axis == last))
                 for axis, n in enumerate(shape))


def _pad_into(chi, pad_before, pad_after, fill, dtype=None):
    """Allocate the padded array once and write the source into it.

    ``np.pad`` allocates a second full-size array; at ~14 GB per production
    volume that transient doubling is what puts the run out of reach.
    """
    dtype = dtype if dtype is not None else chi.dtype
    shape = tuple(n + b + a for n, b, a in zip(chi.shape, pad_before, pad_after))
    out = np.full(shape, fill, dtype=dtype)
    out[tuple(slice(b, b + n) for b, n in zip(pad_before, chi.shape))] = chi
    return out


def _pad_susceptibility(chi, buffer, mode):
    """Legacy padding, factored out unchanged.

    NOTE: mode='constant' pads with numpy's default 0.0, NOT with air. That is
    the pre-existing behaviour the golden fixtures record, so it must not be
    "helpfully" changed here - --pad-value applies only to the zerofill path.
    """
    if mode == 'b0SimISMRM':
        chi = np.pad(chi, ((buffer, buffer), (0, buffer), (buffer, buffer)), mode='edge')
        chi = np.pad(chi, ((0, 0), (buffer, 0), (0, 0)), mode='constant', constant_values=0.35)
    else:
        chi = np.pad(chi, buffer, mode=mode)
    return chi


def compute_bz(susceptibility_distribution, image_resolution=np.array([1, 1, 1]),
               buffer=None, mode='edge', zerofill=None, pad_value=0.35,
               dtype=np.float64, workers=-1):
    """
    Compute the Bz field variation in ppm from a susceptibility distribution,
    using a Fourier-based method.

    Numerically identical to :func:`compute_bz_reference`; the savings are
    in how the arrays are built and held:

    1. Real-input FFT. The susceptibility map is real, so its spectrum is
       Hermitian-symmetric and the second half carries no information. The
       dipole kernel is real and even in k, so multiplying the kept half
       preserves that symmetry and ``irfftn`` reconstructs the real field
       exactly - at half the frequency-domain memory.
    2. k-axes built directly in FFT order with ``fftfreq``/``rfftfreq``, so the
       meshgrid and the ``ifftshift`` pair disappear.
    3. The kernel is applied slab-wise along axis 0 and never materialised whole.
    4. Padding is allocated once rather than via ``np.pad``.
    5. ``scipy.fft`` with threading; ``np.fft`` is single-threaded, which is
       what made the transform look far more expensive than it is.

    Args:
        susceptibility_distribution (numpy.ndarray): 3D susceptibility map, ppm.
        image_resolution (numpy.ndarray, optional): voxel size per axis.
        buffer (int, optional): padding in voxels per side. Defaults to 50 when
            ``zerofill`` is not given. Mutually exclusive with ``zerofill``.
        mode (str, optional): padding mode for the buffer path. Defaults 'edge'.
        zerofill (float, optional): pad each axis to at least this multiple of
            its length (2 = double each dimension), with ``pad_value``. This is
            the mode that treats the object as finite and enclosed in air.
        pad_value (float, optional): susceptibility of the padded region, ppm.
            Used only by the zerofill path. Defaults 0.35 (air).
        dtype (type, optional): working precision.
        workers (int, optional): threads for the FFT. -1 uses every core, which
            is right for a single job but oversubscribes when several subjects
            run concurrently - the run driver sets cores/concurrency instead.

    Returns:
        numpy.ndarray: the field, cropped back to the input shape.
    """
    if zerofill is not None and buffer is not None:
        raise ValueError("buffer and zerofill are mutually exclusive: pass one or the other")

    shape = susceptibility_distribution.shape

    if zerofill is not None:
        if zerofill < 1:
            raise ValueError(f"zerofill must be >= 1, got {zerofill}")
        # rfftn transforms the LAST axis as real and the others as complex, so
        # only that axis wants the real-FFT length table.
        padded = zerofill_padded_shape(shape, zerofill)
        pad_before = tuple((p - n) // 2 for p, n in zip(padded, shape))
        pad_after = tuple(p - n - b for p, n, b in zip(padded, shape, pad_before))
        chi = _pad_into(susceptibility_distribution, pad_before, pad_after,
                        pad_value, dtype=dtype)
        crop = tuple(slice(b, b + n) for b, n in zip(pad_before, shape))
    else:
        buffer = 50 if buffer is None else buffer
        chi = _pad_susceptibility(susceptibility_distribution, buffer, mode)
        chi = chi.astype(dtype, copy=False)
        crop = ((slice(None),) * 3 if buffer == 0
                else (slice(buffer, -buffer),) * 3)

    n0, n1, n2 = (int(n) for n in chi.shape)

    # k-axes already in FFT order: fftfreq(n, d) reproduces
    # ifftshift(linspace(-kmax, kmax, n)) for both parities.
    kx = np.fft.fftfreq(n0, d=float(image_resolution[0])).astype(dtype)
    ky2 = np.fft.fftfreq(n1, d=float(image_resolution[1])).astype(dtype) ** 2
    kz2 = np.fft.rfftfreq(n2, d=float(image_resolution[2])).astype(dtype) ** 2

    spectrum = sfft.rfftn(chi, workers=workers)
    del chi

    # Apply 1/3 - kz^2/k^2 slab by slab. Materialising the kernel whole would
    # cost another ~7 GB on a production volume.
    for i in range(n0):
        k2 = kx[i] * kx[i] + ky2[:, None] + kz2[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            slab = np.divide(kz2[None, :], k2)
        np.subtract(1.0 / 3.0, slab, out=slab)
        if i == 0:
            slab[0, 0] = 1.0 / 3.0          # DC term, undetermined at k=0
        spectrum[i] *= slab

    volume_with_buffer = sfft.irfftn(spectrum, s=(n0, n1, n2), workers=workers)
    del spectrum

    # irfftn cannot infer an odd final axis from a half spectrum; without s= it
    # silently returns the even length. Assert rather than trust.
    assert volume_with_buffer.shape == (n0, n1, n2), (
        f"irfftn returned {volume_with_buffer.shape}, expected {(n0, n1, n2)} - "
        "s= was not honoured")

    return volume_with_buffer[crop]


def save_to_nifti(data, affine_matrix, output_path):
    """
    Save data to NIfTI format to a specified output path.

    Args:
        data (np.array): the data to save
        affine_matrix (np.array): the affine matrix of the data
        output_path (str): the output path to save the file

    Returns:
        None
    """
    data=data.astype(np.float32)
    nifti_image = nib.Nifti1Image(data, affine_matrix)
    nib.save(nifti_image, output_path)



@click.command(help="Compute the magnetic field variation in ppm from a susceptibility distribution in NIfTI format.")
@click.option('-i','--input','input_file', type=click.Path(exists=True), required=True,
              help="Input susceptibility distribution, supported extensions: .nii, .nii.gz")
@click.option('-o', '--output', 'output_file', type=click.Path(), default='fieldmap.nii.gz',
              help="Output fieldmap, supported extensions: .nii, .nii.gz")
@click.option('-b', '--buffer', 'buffer', type=int, default=None, required=False,
                help="Buffer size (voxels) for the k-space grid on each side. "
                     "Defaults to 50. Mutually exclusive with -r/--zerofill.")
@click.option('-m', '--mode', 'mode', type=str, default='edge', required=False,
                help="Padding mode for the susceptibility distribution")
@click.option('-r', '--zerofill', 'zerofill', type=float, default=None, required=False,
                help="Zero-fill ratio: pad each axis to at least this multiple of "
                     "its length (2 = double each dimension), filling with "
                     "--pad-value. Use this when the object is finite and enclosed "
                     "in air. Mutually exclusive with -b/--buffer.")
@click.option('--pad-value', 'pad_value', type=float, default=0.35, required=False,
                help="Susceptibility of the padded region in ppm, used by "
                     "-r/--zerofill. Default 0.35 (air).")
@click.option('--dtype', 'dtype_name', type=click.Choice(['float32', 'float64']),
                default='float32', required=False,
                help="Working precision. float32 halves peak memory; measured "
                     "deviation 4.9e-7 ppm on a 14.8 ppm field.")
@click.option('--workers', 'workers', type=int, default=-1, required=False,
                help="FFT threads. -1 (default) uses every core. Set this to "
                     "cores/concurrency when running several subjects at once, "
                     "or the jobs oversubscribe the machine.")
def compute_fieldmap(input_file, output_file, buffer, mode, zerofill, pad_value,
                     dtype_name, workers):
    """
    Main procedure for performing the simulation.

    Args:
        input_file (str): Path to the susceptibility distribution in NIfTI format.
        output_file (str): Path for the computed fieldmap in NIfTI format.

    Returns:
        None
    """
    if is_nifti(input_file):
        start_time = perf_counter()
        print('Start')
        susceptibility_distribution, image_resolution, affine_matrix = load_sus_dist(input_file)
        print('Susceptibility distribution loaded')
        dtype = np.float32 if dtype_name == 'float32' else np.float64

        if zerofill is not None:
            padded = zerofill_padded_shape(susceptibility_distribution.shape, zerofill)
            n_padded = int(np.prod(padded))
            itemsize = np.dtype(dtype).itemsize
            half = padded[0] * padded[1] * (padded[2] // 2 + 1)
            print(f'Zero-fill R={zerofill}: {susceptibility_distribution.shape} -> {padded} '
                  f'({n_padded/1e9:.2f}e9 voxels), padding with {pad_value} ppm')
            print(f'  approx arrays: real {n_padded*itemsize/2**30:.1f} GB + '
                  f'half-spectrum {half*itemsize*2/2**30:.1f} GB')

        fieldmap = compute_bz(susceptibility_distribution, image_resolution,
                              buffer=buffer, mode=mode, zerofill=zerofill,
                              pad_value=pad_value, dtype=dtype, workers=workers)
        print('Fieldmap simulated')

        # Check if all subdirectories exist and create them if not
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_to_nifti(fieldmap, affine_matrix, output_file)
        print('Saving to NIfTI format')
        end_time = perf_counter()
        print(f'End. Runtime: {end_time-start_time:.2f} seconds')
    else:
        print("The input file must be NIfTI.")

    
