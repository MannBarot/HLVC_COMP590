import numpy as np

bits = np.load('Music_Video_com_fast_PSNR_1024/bits.npy')
quality = np.load('Music_Video_com_fast_PSNR_1024/quality.npy')

print("Bitrate (BR) per frame:", np.round(bits, 2))
print("Quality per frame:", np.round(quality, 2))
print("Average BR:", round(bits.mean(), 2))