import numpy as np
import tensorflow as tf

import vgg16
import utils
import os
from tqdm import tqdm 
from scipy import misc
# from pathlib import Path

# pathlist = Path('../../3D-R2N2/ShapeNet/ShapeNetRendering/').glob('**/*00.png')
# os.path.dirname(str(path))
path = '../../chair_4.0_databset/'
# input_Sym = utils.load_image_com(path + 'chair_rotation_00_only_00.png')[:,:,:3]
# for i in range(1,1000):
# 	tmp_image = utils.load_image_com(path + 'chair_rotation_00_only_%.2d.png' % i)[:,:,:3]
# 	input_Sym = np.concatenate((input_Sym,tmp_image),axis = -1)
# input_Sym = np.load('/home/gaopeng/dict_image.npy')

def generator(path):
	# for j in range(50):
	for i in range(1000):
		# receive input_a
		# input_a = utils.load_image(path + 'chair_rotation_%.2d/chair_rotation_%.2d.png' % (j, j))
		# input_a = utils.load_image('../../chair.png')
		# input_a = input_a[:,:,:3]
		# input_a = input_a.reshape((-1))
		# input_a = np.tile(input_a, 10)
		# input_a = input_a.reshape((10, 224, 224, 3))
		input_a = utils.load_image_com(path + 'chair_source/chair_rotation_source_%.2d.png' % (i*1))[:,:,:3]
		for k in range(1,10,1):
			# input_c = np.concatenate((input_c, utils.load_image_com(path + 'chair_rotation_%.2d/chair_rotation_%.2d_trans_%.2d.png' % (j, j, (i*8 + k)))[:,:,:3]), axis = 0)
			input_a = np.concatenate((input_a, utils.load_image_com(path + 'chair_source/chair_rotation_source_%.2d.png' % (i*10+k))[:,:,:3]), axis = 0)
		input_a = np.reshape(input_a, (10, 224, 224, 3))
		# receive input_b
		# input_b = np.loadtxt(path + 'chair_rotation_%.2d/rendering_metadata_rotation_new_%.2d.txt' % (j, j))
		input_b = np.loadtxt(path + 'rendering_destination_rotation.txt')
		# input_b = (input_b + 1.5)/3.0
		input_b = input_b[(i*10):(i*10+10),:]
		# receive input_c
		# input_c = utils.load_image_com(path + 'chair_rotation_%.2d/chair_rotation_%.2d_trans_%.2d.png' % (j, j, i*8))[:,:,:3]
		input_c = utils.load_image_com(path + 'chair_destination/chair_rotation_destination_%.2d.png' % (i*1))[:,:,:3]
		for k in range(1,10,1):
			# input_c = np.concatenate((input_c, utils.load_image_com(path + 'chair_rotation_%.2d/chair_rotation_%.2d_trans_%.2d.png' % (j, j, (i*8 + k)))[:,:,:3]), axis = 0)
			input_c = np.concatenate((input_c, utils.load_image_com(path + 'chair_destination/chair_rotation_destination_%.2d.png' % (i*10+k))[:,:,:3]), axis = 0)
		input_c = np.reshape(input_c, (10, 224, 224, 3))
		# generate data
		yield (input_a, input_b, input_c)
# input_a = utils.load_image("")
# input_b = dataset from quaternions txt
# input_c = load image

# input_a = np.random.rand(6, 224, 224, 3)
# input_b = np.random.rand(6, 4)
# input_c = np.random.rand(6,137, 137, 3)

# define resnet_block
def residual_block(inputs_layer, nb_blocks, out_filters, strides):
	for _ in range(nb_blocks):
		shortcut = inputs_layer
		# resnet_1_layer
		inputs_layer = tf.layers.conv2d_transpose(inputs = inputs_layer, filters = out_filters, kernel_size = 3, strides = strides, padding='same', data_format='channels_last')
		inputs_layer = tf.contrib.layers.batch_norm(inputs_layer)
		inputs_layer = tf.nn.relu(inputs_layer)
		# resnet_2_layer
		inputs_layer = tf.layers.conv2d_transpose(inputs = inputs_layer, filters = out_filters, kernel_size = 3, strides = 1, padding='same', data_format='channels_last')
		inputs_layer = tf.contrib.layers.batch_norm(inputs_layer)
		inputs_layer = tf.nn.relu(inputs_layer)
		# process shortcut
		shortcut = tf.layers.conv2d_transpose(inputs = shortcut, filters = out_filters, kernel_size = 3, strides = strides, padding='same', data_format='channels_last')
		# skip connection
		inputs_layer += shortcut
	# output resnet result
	return inputs_layer

def VGG16_Synthesis_Network(input_img, input_Semantic):
	#block_1
	# img encoder
	vgg_1 = vgg16.Vgg16()
	vgg_1.build(input_img)
	img_layer_1 = tf.contrib.layers.flatten(vgg_1.pool5)
	img_layer_1 = tf.contrib.layers.fully_connected(img_layer_1, 1024)

	# Semantic encoder
	# Semantic_layer_1 = tf.contrib.layers.fully_connected(input_Semantic, 512)

	# Semantic synthesis
	# z_1 = tf.concat([img_layer_1, Semantic_layer_1], -1)

	# img decoder
	img_layer_1 = tf.expand_dims(img_layer_1,1)
	img_layer_1 = tf.expand_dims(img_layer_1,1)
	# output_1 = tf.layers.conv2d_transpose(inputs = z, filters = 512, kernel_size = 3, strides = (2,2), padding='same', data_format='channels_last', activation = tf.nn.relu)
	# output_2 = tf.layers.conv2d_transpose(inputs = output_1, filters = 256, kernel_size = 3, strides = (2,2), padding='same', data_format='channels_last', activation = tf.nn.relu)
	# output_3 = tf.layers.conv2d_transpose(inputs = output_2, filters = 256, kernel_size = 3, strides = (2,2), padding='same', data_format='channels_last', activation = tf.nn.relu)
	# output_4 = tf.layers.conv2d_transpose(inputs = output_3, filters = 128, kernel_size = 3, strides = (2,2), padding='same', data_format='channels_last', activation = tf.nn.relu)
	# output_5 = tf.layers.conv2d_transpose(inputs = output_4, filters = 64, kernel_size = 3, strides = (2,2), padding='same', data_format='channels_last', activation = tf.nn.relu)
	# output_6 = tf.layers.conv2d_transpose(inputs = output_5, filters = 32, kernel_size = 3, strides = (2,2), padding='same', data_format='channels_last', activation = tf.nn.relu)
	# output_7 = tf.layers.conv2d_transpose(inputs = output_6, filters = 3, kernel_size = 3, strides = (2,2), padding='same', data_format='channels_last', activation = tf.nn.relu)
	output_1_1 = residual_block(inputs_layer = img_layer_1, nb_blocks = 1, out_filters = 512, strides = 2)
	Semantic_layer_1_1 = tf.contrib.layers.fully_connected(input_Semantic, 1024)
	Semantic_layer_1_1 = tf.expand_dims(Semantic_layer_1_1,1)
	Semantic_layer_1_1 = tf.expand_dims(Semantic_layer_1_1,1)
	Semantic_layer_1_1 = tf.layers.conv2d_transpose(inputs = Semantic_layer_1_1, filters = 512, kernel_size = 3, strides = (2,2), padding='same', data_format='channels_last', activation = tf.nn.relu)
	output_1_1 = tf.concat([output_1_1, Semantic_layer_1_1], -1)
	#output_1_1 = tf.layers.conv2d(inputs = output_1_1, filters = 512, kernel_size = 3, strides = (1,1), padding='same', data_format='channels_last', activation = tf.nn.relu)
	output_1_2 = residual_block(inputs_layer = output_1_1, nb_blocks = 1, out_filters = 512, strides = 1)
	Semantic_layer_1_2 = tf.layers.conv2d_transpose(inputs = Semantic_layer_1_1, filters = 256, kernel_size = 3, strides = (1,1), padding='same', data_format='channels_last', activation = tf.nn.relu)
	output_1_2 = tf.concat([output_1_2, Semantic_layer_1_2], -1)
	#output_1_2 = tf.layers.conv2d(inputs = output_1_2, filters = 256, kernel_size = 3, strides = (1,1), padding='same', data_format='channels_last', activation = tf.nn.relu)
	output_1_3 = residual_block(inputs_layer = output_1_2, nb_blocks = 1, out_filters = 256, strides = 2)
	Semantic_layer_1_3 = tf.layers.conv2d_transpose(inputs = Semantic_layer_1_2, filters = 256, kernel_size = 3, strides = (2,2), padding='same', data_format='channels_last', activation = tf.nn.relu)
	output_1_3 = tf.concat([output_1_3, Semantic_layer_1_3], -1)
	#output_1_3 = tf.layers.conv2d(inputs = output_1_3, filters = 256, kernel_size = 3, strides = (1,1), padding='same', data_format='channels_last', activation = tf.nn.relu)
	output_1_4 = residual_block(inputs_layer = output_1_3, nb_blocks = 1, out_filters = 256, strides = 1)
	Semantic_layer_1_4 = tf.layers.conv2d_transpose(inputs = Semantic_layer_1_3, filters = 128, kernel_size = 3, strides = (1,1), padding='same', data_format='channels_last', activation = tf.nn.relu)
	output_1_4 = tf.concat([output_1_4, Semantic_layer_1_4], -1)
	#output_1_4 = tf.layers.conv2d(inputs = output_1_4, filters = 128, kernel_size = 3, strides = (1,1), padding='same', data_format='channels_last', activation = tf.nn.relu)
	output_1_5 = residual_block(inputs_layer = output_1_4, nb_blocks = 1, out_filters = 128, strides = 2)
	Semantic_layer_1_5 = tf.layers.conv2d_transpose(inputs = Semantic_layer_1_4, filters = 128, kernel_size = 3, strides = (2,2), padding='same', data_format='channels_last', activation = tf.nn.relu)
	output_1_5 = tf.concat([output_1_5, Semantic_layer_1_5], -1)
	#output_1_5 = tf.layers.conv2d(inputs = output_1_5, filters = 64, kernel_size = 3, strides = (1,1), padding='same', data_format='channels_last', activation = tf.nn.relu)
	output_1_6 = residual_block(inputs_layer = output_1_5, nb_blocks = 1, out_filters = 128, strides = 1)
	Semantic_layer_1_6 = tf.layers.conv2d_transpose(inputs = Semantic_layer_1_5, filters = 64, kernel_size = 3, strides = (1,1), padding='same', data_format='channels_last', activation = tf.nn.relu)
	output_1_6 = tf.concat([output_1_6, Semantic_layer_1_6], -1)
	#output_1_6 = tf.layers.conv2d(inputs = output_1_6, filters = 32, kernel_size = 3, strides = (1,1), padding='same', data_format='channels_last', activation = tf.nn.relu)
	output_1_7 = residual_block(inputs_layer = output_1_6, nb_blocks = 1, out_filters = 64, strides = 2)
	Semantic_layer_1_7 = tf.layers.conv2d_transpose(inputs = Semantic_layer_1_6, filters = 64, kernel_size = 3, strides = (2,2), padding='same', data_format='channels_last', activation = tf.nn.relu)
	output_1_7 = tf.concat([output_1_7, Semantic_layer_1_7], -1)
	#output_1_7 = tf.layers.conv2d(inputs = output_1_7, filters = 3, kernel_size = 3, strides = (1,1), padding='same', data_format='channels_last', activation = tf.nn.relu)
	output_1_8 = residual_block(inputs_layer = output_1_7, nb_blocks = 1, out_filters = 64, strides = 1)
	Semantic_layer_1_8 = tf.layers.conv2d_transpose(inputs = Semantic_layer_1_7, filters = 32, kernel_size = 3, strides = (1,1), padding='same', data_format='channels_last', activation = tf.nn.relu)
	output_1_8 = tf.concat([output_1_8, Semantic_layer_1_8], -1)

	output_1_9 = residual_block(inputs_layer = output_1_8, nb_blocks = 1, out_filters = 32, strides = 2)
	Semantic_layer_1_9 = tf.layers.conv2d_transpose(inputs = Semantic_layer_1_8, filters = 32, kernel_size = 3, strides = (2,2), padding='same', data_format='channels_last', activation = tf.nn.relu)
	output_1_9 = tf.concat([output_1_9, Semantic_layer_1_9], -1)

	output_1_10 = residual_block(inputs_layer = output_1_9, nb_blocks = 1, out_filters = 32, strides = 1)
	Semantic_layer_1_10 = tf.layers.conv2d_transpose(inputs = Semantic_layer_1_9, filters = 16, kernel_size = 3, strides = (1,1), padding='same', data_format='channels_last', activation = tf.nn.relu)
	output_1_10 = tf.concat([output_1_10, Semantic_layer_1_10], -1)

	output_1_11 = residual_block(inputs_layer = output_1_10, nb_blocks = 1, out_filters = 16, strides = 2)
	Semantic_layer_1_11 = tf.layers.conv2d_transpose(inputs = Semantic_layer_1_10, filters = 16, kernel_size = 3, strides = (2,2), padding='same', data_format='channels_last', activation = tf.nn.relu)
	output_1_11 = tf.concat([output_1_11, Semantic_layer_1_11], -1)

	output_1_12 = residual_block(inputs_layer = output_1_11, nb_blocks = 1, out_filters = 16, strides = 1)
	Semantic_layer_1_12 = tf.layers.conv2d_transpose(inputs = Semantic_layer_1_11, filters = 8, kernel_size = 3, strides = (1,1), padding='same', data_format='channels_last', activation = tf.nn.relu)
	output_1_12 = tf.concat([output_1_12, Semantic_layer_1_12], -1)

	output_1_13 = residual_block(inputs_layer = output_1_12, nb_blocks = 1, out_filters = 8, strides = 2)
	Semantic_layer_1_13 = tf.layers.conv2d_transpose(inputs = Semantic_layer_1_12, filters = 8, kernel_size = 3, strides = (2,2), padding='same', data_format='channels_last', activation = tf.nn.relu)
	output_1_13 = tf.concat([output_1_13, Semantic_layer_1_13], -1)

	output_1_14 = residual_block(inputs_layer = output_1_13, nb_blocks = 1, out_filters = 8, strides = 1)
	Semantic_layer_1_14 = tf.layers.conv2d_transpose(inputs = Semantic_layer_1_13, filters = 4, kernel_size = 3, strides = (1,1), padding='same', data_format='channels_last', activation = tf.nn.relu)
	output_1_14 = tf.concat([output_1_14, Semantic_layer_1_14], -1)

	output_1_15 = tf.layers.conv2d_transpose(inputs = output_1_14, filters = 3, kernel_size = 3, strides = (1,1), padding='same', data_format='channels_last', activation = tf.nn.relu)

	# output
	output_1_16 = tf.image.resize_images(output_1_15, [224,224])

	#block_2
	# img encoder
	# vgg_2 = vgg16.Vgg16()
	# vgg_2.build(output_1_8)
	# img_layer_2 = tf.contrib.layers.flatten(vgg_2.pool5)
	# img_layer_2 = tf.contrib.layers.fully_connected(img_layer_2, 256)

	# # Semantic encoder
	# Semantic_layer_2 = tf.contrib.layers.fully_connected(input_Semantic, 256)

	# # Semantic synthesis
	# z_2 = tf.concat([img_layer_2, Semantic_layer_2], -1)
	# z_2 = tf.concat([z_2, img_layer_1], -1)

	# # img decoder
	# z_2 = tf.expand_dims(z_2,1)
	# z_2 = tf.expand_dims(z_2,1)
	# output_2_1 = residual_block(inputs_layer = z_2, nb_blocks = 1, out_filters = 512, strides = 2)
	# output_2_2 = residual_block(inputs_layer = output_2_1, nb_blocks = 1, out_filters = 256, strides = 2)
	# output_2_3 = residual_block(inputs_layer = output_2_2, nb_blocks = 1, out_filters = 256, strides = 2)
	# output_2_4 = residual_block(inputs_layer = output_2_3, nb_blocks = 1, out_filters = 128, strides = 2)
	# output_2_5 = residual_block(inputs_layer = output_2_4, nb_blocks = 1, out_filters = 64, strides = 2)
	# output_2_6 = residual_block(inputs_layer = output_2_5, nb_blocks = 1, out_filters = 32, strides = 2)
	# output_2_7 = residual_block(inputs_layer = output_2_6, nb_blocks = 1, out_filters = 3, strides = 2)
	# # output
	# output_2_8 = tf.image.resize_images(output_2_7, [224,224])
	return output_1_16

# placeholder of networks
input_img = tf.placeholder(tf.float32, [None, 224, 224, 3])
input_Semantic = tf.placeholder(tf.float32, [None, 6])
output_img = tf.placeholder(tf.float32, [None, 224, 224, 3])

# output of networks
model = VGG16_Synthesis_Network(input_img, input_Semantic)

# loss of networks
with tf.name_scope('cost'):
	loss = tf.reduce_mean(tf.square(model - output_img))
	tf.summary.scalar('loss',loss)

# optimizer of networks
train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

# path of checkpoint
saver = tf.train.Saver()
checkpoint_dir_1 = 'model_weight_rotation_v4.0/'
checkpoint_dir_2 = 'model_weight_rotation_v3.1/'

# GPU using rate
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.9

with tf.Session() as sess:
	# initializer variables
	writer = tf.summary.FileWriter('./logs',sess.graph)
	merged = tf.summary.merge_all()
	sess.run(tf.global_variables_initializer())
	sess.graph.finalize()

	# checkpoint loading
	flag_x = 0
	if(flag_x):
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir_2)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess,ckpt.model_checkpoint_path)

	for epoch in range(50):
		# generate database
		generator_result = generator(path)
		# training model
		for i in tqdm(range(1000)):
			input_a, input_b, input_c = generator_result.__next__()
			sess.run(train_step,feed_dict={input_img:input_a, input_Semantic:input_b, output_img:input_c})
			if ((i + 1) % 50 == 0):
				print([i+1],sess.run(loss, feed_dict={input_img:input_a, input_Semantic:input_b, output_img:input_c}))
		# print loss of epoch
		print([epoch+1],sess.run(loss, feed_dict={input_img:input_a, input_Semantic:input_b, output_img:input_c}))
		# tensorboard
		summary = sess.run(merged,feed_dict = {input_img:input_a, input_Semantic:input_b, output_img:input_c})
		writer.add_summary(summary,epoch)
		# save 5th epoch weight
		if ((epoch+1) % 50 == 0):
			saver.save(sess, checkpoint_dir_1 + 'model_ckpt', global_step=epoch+1)
	# output image for testing
	# for i in range(50):
	# 	# input_image_test
	# 	input_test = utils.load_image('../../' + 'chair.png')
	# 	input_test = input_test[:,:,:3]
	# 	# input_test = input_test.reshape((-1))
	# 	# input_test = np.tile(input_test, 10)
	# 	input_test = input_test.reshape((1, 224, 224, 3))
	# 	# input_Semantic_test
	# 	input_Semantic_test = np.loadtxt('only_rotation_test_example.txt')
	# 	# input_Semantic_test = np.loadtxt(path + 'rendering_metadata_rotation_00.txt')
	# 	# input_Semantic_test = input_Semantic_test.reshape((1,6))
	# 	input_Semantic_test = input_Semantic_test[(i*1):(i*1+1),:]
	# 	# input_c_test
	# 	input_c_test = utils.load_image_com('../../chair_only_rotation_test_true/chair_only_rotation_true/' + 'chair_rotation_00_only_%.2d.png' % i)[:,:,:3]
	# 	# input_c_test = utils.load_image_com('../../chair_only_rotation_test_true/chair_only_rotation_true/' + 'chair_rotation_00_only_%.2d.png' % (i*10))[:,:,:3]
	# 	# for k in range(1,10,1):
	# 	# 	# input_c = np.concatenate((input_c, utils.load_image_com(path + 'chair_rotation_%.2d/chair_rotation_%.2d_trans_%.2d.png' % (j, j, (i*8 + k)))[:,:,:3]), axis = 0)
	# 	# 	input_c_test = np.concatenate((input_c_test, utils.load_image_com('../../chair_only_rotation_test_true/chair_only_rotation_true/' + 'chair_rotation_00_only_%.2d.png' % (i*10+k))[:,:,:3]), axis = 0)
	# 	# input_c_test = np.reshape(input_c_test, (10, 224, 224, 3))
	# 	input_c_test = input_c_test.reshape((1,224,224,3))
	# 	# plot output_model
	# 	output_test_image = sess.run(model,feed_dict={input_img:input_test, input_Semantic:input_Semantic_test})
	# 	output_test_image = output_test_image.reshape((224, 224, 3))
	# 	print([i+1],sess.run(loss, feed_dict={input_img:input_test, input_Semantic:input_Semantic_test, output_img:input_c_test}))
	# 	# misc.imsave('../../chairr1_test_test.png' , output_test_image)
	# 	misc.imsave('../../chair_only_rotation_test_true/chair_only_rotation_test/chair_only_rotation_test_%.2d.png' % i , output_test_image)