import pytest
import random
import numpy as np
import os
from bluestein_fft import chunk_out_zoom_fft_2d, chunk_out_zoom_fft_2d_mod
#Test the Bluestein chunk out functions

# Centered data
def gen_param():
    N_x = random.randrange(7, 200, 2)
    N_out = random.randrange(7, 200, 2)
    N_X = random.randrange(max(N_x, N_out) + 4, 800, 2)
    Z_pad = (N_X - 1) / N_x 
    while (Z_pad*N_x) + 1 != N_X: 
        N_X = random.randrange(max(N_x, N_out) + 4, 800, 2)
        Z_pad = (N_X - 1) / N_x 
    N_chirp = N_x + N_out - 1
    N_chunk = random.randrange(2, N_out//2)
    return N_x, N_X, N_out, Z_pad, N_chirp, N_chunk

def sim_data_square(N_x, N_X):
    x = np.zeros((N_X, N_X))
    x[N_X//2 - N_x//2 : N_X//2 + N_x//2 + 1 , N_X//2 - N_x//2 : N_X//2 + N_x//2 + 1 ] = np.ones((N_x,N_x))
    x[N_X//2 - 2: N_X //2, N_X//2 + 1]*=random.uniform(.2, 10)
    return x

def gen_data():
    params = gen_param()
    x = sim_data_square(params[0], params[1])
    return params[0], params[1], params[2], params[3], params[4], params[5], x

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
    (gen_data()),
]

@pytest.mark.parametrize("N_x, N_X, N_out, Z_pad, N_chirp, N_chunk, x", test_data)
def test_chunk_out_bluestein_fft(N_x, N_X, N_out, Z_pad, N_chirp, N_chunk, x):
    # Test bluestein FFT chunk out functions with numpy FFT2
    print (N_x, N_X, N_out, N_chunk, np.shape(x))
    real_ft_x_shift = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))\
    [N_X//2 - N_out//2 : N_X//2 + N_out//2 + 1, N_X//2 - N_out//2 : N_X//2 + N_out//2 + 1]
    real_ft_x = np.fft.fftshift(np.fft.fft2(x))\
    [N_X//2 - N_out//2 : N_X//2 + N_out//2 + 1, N_X//2 - N_out//2 : N_X//2 + N_out//2 + 1]
    for i in range(N_chunk):
        for j in range(N_chunk):
            sec_N_x = sec_N_y = N_out//N_chunk
            if i == N_chunk-1: sec_N_x = N_out - i*(N_out//N_chunk)
            if j == N_chunk-1: sec_N_y = N_out - j*(N_out//N_chunk)
            chunk_ft_x_shift = chunk_out_zoom_fft_2d_mod(x, N_x, N_out_x = sec_N_x, N_out_y = sec_N_y, start_chunk_x = i*(N_out//N_chunk) - (N_out//2), start_chunk_y = j*(N_out//N_chunk) - (N_out//2), N_X=N_X)
            chunk_ft_x = chunk_out_zoom_fft_2d(x, N_x, N_out_x = sec_N_x, N_out_y = sec_N_y, start_chunk_x = i*(N_out//N_chunk) - (N_out//2), start_chunk_y = j*(N_out//N_chunk) - (N_out//2), N_X=N_X)
            assert np.allclose(chunk_ft_x_shift, real_ft_x_shift[i*(N_out//N_chunk): i*(N_out//N_chunk) + sec_N_x, j*(N_out//N_chunk): j*(N_out//N_chunk) + sec_N_y])
            assert np.allclose(chunk_ft_x, real_ft_x[i*(N_out//N_chunk): i*(N_out//N_chunk) + sec_N_x, j*(N_out//N_chunk): j*(N_out//N_chunk) + sec_N_y])
