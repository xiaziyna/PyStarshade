import pytest
import random
import numpy as np
from pystarshade.diffraction.util import bluestein_pad, trunc_2d, N_in_2d
from pystarshade.diffraction.bluestein_fft import zoom_fft_2d_mod, zoom_fft_2d, wrap_chunk_fft

# Centered data
def gen_param():
    N_x = random.randrange(3, 200, 2)
    N_out = random.randrange(3, 200, 2)
    N_X = random.randrange(max(N_x, N_out), 800, 2)
    Z_pad = (N_X - 1) / N_x 
    while (Z_pad*N_x) + 1 != N_X: 
        N_X = random.randrange(max(N_x, N_out), 800, 2)
        Z_pad = (N_X - 1) / N_x 
    N_chirp = N_x + N_out - 1
    return N_x, N_X, N_out, Z_pad, N_chirp

def sim_data_square(N_x, N_X):
    x = np.zeros((N_X, N_X))
    x[N_X//2 - N_x//2 : N_X//2 + N_x//2 + 1 , N_X//2 - N_x//2 : N_X//2 + N_x//2 + 1 ] = np.ones((N_x,N_x))
    x[N_X//2 - 2: N_X //2, N_X//2 + 1]*=random.uniform(.2, 10)
    return x

def gen_data():
    params = gen_param()
    x = sim_data_square(params[0], params[1])
    return params[0], params[1], params[2], params[3], params[4], x

def gen_data_1():
    x = sim_data_square(5, 21)
    return 5, 21, 5, 4, 9, x

def gen_data_2():
    x = sim_data_square(5, 5)
    return 5, 5, 5, 4/5, 9, x

# Sample data for testing
test_data = [
    (gen_data()),
    (gen_data()),
    (gen_data()),
    (gen_data()),
    (gen_data()),
    (gen_data()),
    (gen_data()),
    (gen_data()),
    (gen_data_1()),
    (gen_data_2())
]

@pytest.mark.parametrize("N_x, N_X, N_out, Z_pad, N_chirp, x", test_data)
def test_bluestein_fft(N_x, N_X, N_out, Z_pad, N_chirp, x):
    print (N_x, N_X, N_out, np.shape(x))
    # Test bluestein FFT functions with numpy FFT2
    blue_ft_x_shift = zoom_fft_2d_mod(x, N_x, N_out, Z_pad=Z_pad)
    blue_ft_x_shift_new = zoom_fft_2d_mod(x, N_x, N_out, N_X=N_X)
    real_ft_x_shift = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))\
    [N_X//2 - N_out//2: N_X//2 + N_out//2 + 1, N_X//2 - N_out//2: N_X//2 + N_out//2 + 1]
    
    blue_ft_x = zoom_fft_2d(x, N_x, N_out, Z_pad=Z_pad)
    blue_ft_x_new = zoom_fft_2d(x, N_x, N_out, N_X=N_X)
    real_ft_x = np.fft.fftshift(np.fft.fft2(x))\
    [N_X//2 - N_out//2: N_X//2 + N_out//2 + 1, N_X//2 - N_out//2: N_X//2 + N_out//2 + 1]

    assert np.allclose(blue_ft_x_shift, real_ft_x_shift)
    assert np.allclose(blue_ft_x_shift_new, blue_ft_x_shift)
    assert np.allclose(blue_ft_x, real_ft_x)
    assert np.allclose(blue_ft_x_new, blue_ft_x)


@pytest.mark.parametrize("N_x, N_X, N_out, Z_pad, N_chirp, x", test_data)
def test_chunked_bluestein_fft(N_x, N_X, N_out, Z_pad, N_chirp, x):
    # Test chunked FFT in a compact way
    print (N_x, N_X, N_out)
    test_chunked_bs = wrap_chunk_fft(x, N_x, N_out, N_X, mod=0)
    real_ft_x = np.fft.fftshift(np.fft.fft2(x))[N_X//2 - N_out//2: N_X//2 + N_out//2 + 1, N_X//2 - N_out//2: N_X//2 + N_out//2 + 1]
    assert np.allclose(test_chunked_bs, real_ft_x)


@pytest.mark.parametrize("N_x, N_X, N_out, Z_pad, N_chirp, x", test_data)
def test_chunked_bluestein_fft_verbose(N_x, N_X, N_out, Z_pad, N_chirp, x):
    # Chunking the input into 4 smaller segments
    # This verbose test is for debugging
    # len_seg = [[0, 0], [0, -1], [-1, -1], [-1, 0]]
    x_trunc = trunc_2d(x, N_x)
    in_N_X, in_N_Y = np.shape(x) # input shape
    bit_x = in_N_X%2 # checks if even or odd length
    bit_y = in_N_Y%2

    # For real FFT
    x_1 = x[:in_N_X//2 + bit_x, :in_N_Y//2 + bit_y] #upper left
    x_2 = x[:in_N_X//2 + bit_x, in_N_Y//2 + bit_y:] #upper right
    x_3 = x[in_N_X//2 + bit_x:, in_N_Y//2 + bit_y:] #lower right
    x_4 = x[in_N_X//2 + bit_x:, :in_N_Y//2 + bit_y] #lower left

    # For Bluestein FFT
    x1 = x_trunc[:N_x//2 + bit_x, :N_x//2 + bit_x] #upper left
    x2 = x_trunc[:N_x//2 + bit_x, N_x//2 + bit_x:] #upper right
    x3 = x_trunc[N_x//2 + bit_x:, N_x//2 + bit_x:] #lower right
    x4 = x_trunc[N_x//2 + bit_x:, :N_x//2 + bit_x] #lower left

    shift_bit = 1 - (N_x//2 + 1)%2
    print (N_x//2, shift_bit)
    if shift_bit:
        x1 = np.pad(x1, ((0,1),(0,1)), 'constant')
        x2 = np.pad(x2, ((0,1),(0,2)), 'constant')
        x3 = np.pad(x3, ((0,2),(0,2)), 'constant')
        x4 = np.pad(x4, ((0,2),(0,1)), 'constant')
    else: 
        x2 = np.pad(x2, ((0,0),(0,1)), 'constant')
        x3 = np.pad(x3, ((0,1),(0,1)), 'constant')
        x4 = np.pad(x4, ((0,1),(0,0)), 'constant')

    sec_N_x = N_x//2 + 1 #this is the (max) size of non-zero portion of segment, same for all 4 chunks
    sec_N_x += 1 - sec_N_x%2 # want this to be an odd number as it makes it easier to calculate shifts and center

    # Pad each truncated segment (centered)
    x_1_cent = bluestein_pad(x1, sec_N_x, N_out)
    x_2_cent = bluestein_pad(x2, sec_N_x, N_out)
    x_3_cent = bluestein_pad(x3, sec_N_x, N_out)
    x_4_cent = bluestein_pad(x4, sec_N_x, N_out)

    N_x_seg = in_N_X//2 + bit_x
    N_y_seg = in_N_Y//2 + bit_y
    shift_seg = N_x_seg // 2 - N_x//4

    N_x_1 = max(N_in_2d(x_1))
    N_x_2 = max(N_in_2d(x_2))
    N_x_3 = max(N_in_2d(x_3))
    N_x_4 = max(N_in_2d(x_4))
    N_x_1 += 1 - N_x_1%2
    N_x_2 += 1 - N_x_2%2
    N_x_3 += 1 - N_x_3%2
    N_x_4 += 1 - N_x_4%2
    phase_shift =  (sec_N_x//2)

    zoom_ft_x_1 = zoom_fft_2d(x_1_cent, sec_N_x, N_out, N_X=N_X)
    out_fac = np.exp ( np.arange(-(N_out//2), (N_out//2) + 1) * (1j * 2 * np.pi * (phase_shift - shift_bit) * (1 / (N_X)) ) )

    x_test = np.zeros((N_X, N_X))
    x_test[:N_X//2 + bit_x, :N_X//2 + bit_x] = x_1

    real_ft_x_1 = np.fft.fftshift(np.fft.fft2(x_test))[N_X//2 - N_out//2: N_X//2 + N_out//2 + 1, N_X//2 - N_out//2: N_X//2 + N_out//2 + 1]
    chunk_ft_x_1 = zoom_ft_x_1*np.outer(out_fac, out_fac)

    zoom_ft_x_2 = zoom_fft_2d(x_2_cent, sec_N_x, N_out, N_X=N_X)

    out_fac_1 = np.exp ( np.arange(-(N_out//2), (N_out//2) + 1) * (1j * 2 * np.pi * (phase_shift - shift_bit) * (1 / (N_X)) ) )
    out_fac_2 = np.exp ( np.arange(-(N_out//2), (N_out//2) + 1) * (1j * 2 * np.pi * -(phase_shift + 1) * (1 / (N_X)) ) )

    x_test = np.zeros((N_X, N_X))
    x_test[:N_X//2 + bit_x, N_X//2 + bit_x:] = x_2

    real_ft_x_2 = np.fft.fftshift(np.fft.fft2(x_test))[N_X//2 - N_out//2: N_X//2 + N_out//2 + 1, N_X//2 - N_out//2: N_X//2 + N_out//2 + 1]
    chunk_ft_x_2 = zoom_ft_x_2*np.outer(out_fac_1, out_fac_2)

    zoom_ft_x_3 = zoom_fft_2d(x_3_cent, sec_N_x, N_out, N_X=N_X)

    out_fac_1 = np.exp ( np.arange(-(N_out//2), (N_out//2) + 1) * (1j * 2 * np.pi * -(phase_shift + 1) * (1 / (N_X)) ) )
    out_fac_2 = np.exp ( np.arange(-(N_out//2), (N_out//2) + 1) * (1j * 2 * np.pi * -(phase_shift + 1) * (1 / (N_X)) ) )

    x_test = np.zeros((N_X, N_X))
    x_test[N_X//2 + bit_x:, N_X//2 + bit_x:]= x_3
    real_ft_x_3 = np.fft.fftshift(np.fft.fft2(x_test))[N_X//2 - N_out//2: N_X//2 + N_out//2 + 1, N_X//2 - N_out//2: N_X//2 + N_out//2 + 1]
    chunk_ft_x_3 = zoom_ft_x_3*np.outer(out_fac_1, out_fac_2)

    zoom_ft_x_4 = zoom_fft_2d(x_4_cent, sec_N_x, N_out, N_X=N_X)

    out_fac_1 = np.exp ( np.arange(-(N_out//2), (N_out//2) + 1) * (1j * 2 * np.pi * -(phase_shift + 1) * (1 / (N_X)) ) )
    out_fac_2 = np.exp ( np.arange(-(N_out//2), (N_out//2) + 1) * (1j * 2 * np.pi * (phase_shift - shift_bit) * (1 / (N_X)) ) )

    x_test = np.zeros((N_X, N_X))
    x_test[N_X//2 + bit_x:, :N_X//2 + bit_x]= x_4

    real_ft_x_4 = np.fft.fftshift(np.fft.fft2(x_test))[N_X//2 - N_out//2: N_X//2 + N_out//2 + 1, N_X//2 - N_out//2: N_X//2 + N_out//2 + 1]
    chunk_ft_x_4 = zoom_ft_x_4*np.outer(out_fac_1, out_fac_2)

    test_chunked_bs = wrap_chunk_fft(x, N_x, N_out, N_X, mod=0)
    assert np.allclose(real_ft_x_1, chunk_ft_x_1)
    assert np.allclose(real_ft_x_2, chunk_ft_x_2)
    assert np.allclose(real_ft_x_3, chunk_ft_x_3)
    assert np.allclose(real_ft_x_4, chunk_ft_x_4)
    assert np.allclose(real_ft_x_1 + real_ft_x_2 + real_ft_x_3 + real_ft_x_4, test_chunked_bs)
