import numpy as np

bits = np.load('Music_Video_com_fast_PSNR_1024/bits.npy')
quality = np.load('Music_Video_com_fast_PSNR_1024/quality.npy')

print("\n--- Per-Frame Bitrate (BR/bpp) ---")
print(np.round(bits, 2))

print("\n--- Per-Frame Quality (PSNR in dB) ---")
print(np.round(quality, 2))

print("\n--- Per-Frame Summary (Frame | BR | Quality) ---")
for frame in range(len(bits)):
    print(f"Frame {frame+1:3d}: BR = {bits[frame]:6.2f} bpp | Quality = {quality[frame]:6.2f} dB")

print("=" * 80)
print("HLVC ENCODING/DECODING RESULTS - Music_Video (PSNR mode, lambda=1024)")
print("=" * 80)

print("\n--- BITRATE (BR/bpp) Statistics ---")
print(f"Average BR:        {round(bits.mean(), 2)} bpp")
print(f"Min BR:            {round(bits.min(), 2)} bpp")
print(f"Max BR:            {round(bits.max(), 2)} bpp")
print(f"Std Dev BR:        {round(bits.std(), 2)} bpp")
print(f"Total bits:        {round(bits.sum(), 2)}")

print("\n--- QUALITY (PSNR) Statistics ---")
print(f"Average Quality:   {round(quality.mean(), 2)} dB")
print(f"Min Quality:       {round(quality.min(), 2)} dB")
print(f"Max Quality:       {round(quality.max(), 2)} dB")
print(f"Std Dev Quality:   {round(quality.std(), 2)} dB")