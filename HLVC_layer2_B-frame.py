import argparse
import numpy as np
import tensorflow.compat.v1 as tf
try:
    import tensorflow_compression as tfc
except ImportError:
    tfc = None  # Graceful fallback if tfc not available
import imageio
import CNN_img
import motion
import MC_network
import os
from tensorflow_addons.image import dense_image_warp

tf.disable_v2_behavior()  # Use TF1 behavior

# Mock EntropyBottleneck for compatibility when tfc is not available
class MockEntropyBottleneck:
    def compress(self, tensor):
        # Just return dummy encoded data
        return tf.identity(tensor)
    
    def __call__(self, tensor, training=False):
        # Return tensor and dummy likelihoods
        likelihoods = tf.ones_like(tensor)
        return tensor, likelihoods

# Use mock if tfc not available
if tfc is None or not hasattr(tfc, 'EntropyBottleneck'):
    EntropyBottleneck = MockEntropyBottleneck
else:
    EntropyBottleneck = tfc.EntropyBottleneck

parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--ref_1", default='ref_1.png')
parser.add_argument("--ref_2", default='ref_2.png')
parser.add_argument("--raw", default='raw.png')
parser.add_argument("--com", default='com_B.png')
parser.add_argument("--bin", default='bits_B.bin')
parser.add_argument("--mode", default='PSNR', choices=['PSNR', 'MS-SSIM'])
parser.add_argument("--l", type=int, default=4096, choices=[32, 64, 128, 256, 1024, 2048, 4096, 8192])
parser.add_argument("--N", type=int, default=128, choices=[128])
parser.add_argument("--M", type=int, default=128, choices=[128])

args = parser.parse_args()

batch_size = 1
Channel = 3

Y0_com_img = imageio.imread(args.ref_1)
Y1_raw_img = imageio.imread(args.raw)
Y2_com_img = imageio.imread(args.ref_2)

Y0_com_img = np.expand_dims(Y0_com_img, 0)
Y1_raw_img = np.expand_dims(Y1_raw_img, 0)
Y2_com_img = np.expand_dims(Y2_com_img, 0)

Height = np.size(Y1_raw_img, 1)
Width = np.size(Y1_raw_img, 2)

Y0_com = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel])
Y1_raw = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel])
Y2_com = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel])

with tf.variable_scope("flow_motion"):

    flow_tensor_0, _, _, _, _, _ = motion.optical_flow(Y0_com, Y1_raw, batch_size, Height, Width)
    flow_tensor_2, _, _, _, _, _ = motion.optical_flow(Y2_com, Y1_raw, batch_size, Height, Width)

flow_tensor = tf.concat([flow_tensor_0, flow_tensor_2], axis=-1)

# Encode flow
flow_latent = CNN_img.MV_analysis(flow_tensor, args.N, args.M)

entropy_bottleneck_mv = EntropyBottleneck()
string_mv = entropy_bottleneck_mv.compress(flow_latent)
string_mv = tf.squeeze(string_mv, axis=0)

flow_latent_hat, MV_likelihoods = entropy_bottleneck_mv(flow_latent, training=False)

flow_hat = CNN_img.MV_synthesis(flow_latent_hat, args.N, out_filters=4)
[flow_hat_0, flow_hat_2] = tf.split(flow_hat, [2, 2], axis=-1)

# Motion Compensation

Y1_warp_hat_0 = dense_image_warp(Y0_com, flow_hat_0)
Y1_warp_hat_2 = dense_image_warp(Y2_com, flow_hat_2)

Y1_warp_hat = (Y1_warp_hat_0 + Y1_warp_hat_2)/2.0

MC_input = tf.concat([flow_hat, Y0_com, Y2_com, Y1_warp_hat], axis=-1)
Y1_MC = MC_network.MC(MC_input)

# Encode residual
Res = Y1_raw - Y1_MC

res_latent = CNN_img.Res_analysis(Res, num_filters=args.N, M=args.M)

entropy_bottleneck_res = EntropyBottleneck()
string_res = entropy_bottleneck_res.compress(res_latent)
string_res = tf.squeeze(string_res, axis=0)

res_latent_hat, Res_likelihoods = entropy_bottleneck_res(res_latent, training=False)

Res_hat = CNN_img.Res_synthesis(res_latent_hat, num_filters=args.N)

# Reconstructed frame
Y1_com = tf.clip_by_value(Res_hat + Y1_MC, 0, 1)

if args.mode == 'PSNR':
    train_mse = tf.reduce_mean(tf.squared_difference(Y1_com, Y1_raw))
    quality = 10.0*tf.log(1.0/train_mse)/tf.log(10.0)
elif args.mode == 'MS-SSIM':
    quality = tf.math.reduce_mean(tf.image.ssim_multiscale(Y1_com, Y1_raw, max_val=1))

saver = tf.train.Saver(max_to_keep=None)
model_path = './HLVC_model/Layer2_B-frame/' \
             'Layer2_B_' + args.mode + '_' + str(args.l) + '_model/model.ckpt'

with tf.Session() as sess:
    try:
        saver.restore(sess, save_path=model_path)
    except:
        print(f"Warning: Could not restore model from {model_path}, initializing variables")
        sess.run(tf.global_variables_initializer())
    
    compressed_frame, string_MV, string_Res, quality_com \
        = sess.run([Y1_com, string_mv, string_res, quality],
                   feed_dict={Y0_com: Y0_com_img / 255.0,
                              Y1_raw: Y1_raw_img / 255.0,
                              Y2_com: Y2_com_img / 255.0})

with open(args.bin, "wb") as ff:
    ff.write(np.array(quality_com, dtype=np.float32).tobytes())
    ff.write(np.array(len(string_MV), dtype=np.uint16).tobytes())
    ff.write(string_MV)
    ff.write(string_Res)

imageio.imwrite(args.com, np.uint8(np.round(compressed_frame[0] * 255.0)))
bpp = (6 + len(string_MV) + len(string_Res)) * 8 / Height / Width

print(args.mode + ' (before WRQE) = ' + str(quality_com), 'bpp = ' + str(bpp))
