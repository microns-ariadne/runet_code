import time
import glob
import numpy as np
import scipy.misc as misc
import random
from PIL import Image
import h5py
import pdb
import scipy.ndimage.morphology as morph
import scipy.ndimage as ndimage

ac3_filename = '/n/coxfs01/eric_wu/synapse/data/ac3.npy'
ac4_filename = '/n/coxfs01/eric_wu/synapse/data/ac4.npy'
ecs_train_filename = '/n/coxfs01/eric_wu/synapse/data/ecs.npy'
ecs_test_filename = '/n/coxfs01/eric_wu/synapse/data/ecs_test.npy'
ecs_train_halfres_filename = '/n/coxfs01/eric_wu/synapse/data/ecs_halfres.npy'

ecs_test_reann_filename = '/n/coxfs01/eric_wu/synapse/data/ecs_test_reann.npy'
ecs_test_lee_filename = '/n/coxfs01/eric_wu/synapse/data/ecs_test_lee.npy'


def normalizeImage(img, saturation_level = 0.05):
	if len(np.unique(img)) == 1:
		return img
	sortedValues = np.sort(img.ravel())
	if saturation_level == 0.0:
		minVal = np.float32(sortedValues[0])
		maxVal = np.float32(sortedValues[np.int(len(sortedValues) - 1)])
	else:
		minVal = np.float32(sortedValues[np.int(len(sortedValues) * (saturation_level / 2))])
		maxVal = np.float32(sortedValues[np.int(len(sortedValues) * (1 - saturation_level / 2))])
	normImg = np.float32(img - minVal) * (255 / (maxVal - minVal))
	normImg[normImg < 0] = 0
	normImg[normImg > 255] = 255
	return np.float32(normImg) / 255.0

def normalize(vol):
	return (vol-np.min(vol))/(np.max(vol)-np.min(vol))

def mirror_image_layer(img, cropSize = 92):
	if cropSize == 0:
		return img
	mirror_image = np.zeros((img.shape[0] + 2 * cropSize, img.shape[0] + 2 * cropSize))
	length = img.shape[0]
	mirror_image[cropSize:cropSize + length, cropSize:cropSize + length] = img
	mirror_image[0:cropSize, 0:cropSize] = np.rot90(img[0:cropSize, 0:cropSize], 2)
	mirror_image[-cropSize:, 0:cropSize] = np.rot90(img[-cropSize:, 0:cropSize], 2)
	mirror_image[0:cropSize, -cropSize:] = np.rot90(img[0:cropSize, -cropSize:], 2)
	mirror_image[-cropSize:, -cropSize:] = np.rot90(img[-cropSize:, -cropSize:], 2)
	mirror_image[0:cropSize, cropSize:cropSize + length] = np.flipud(img[0:cropSize, 0:length])
	mirror_image[cropSize:cropSize + length, 0:cropSize] = np.fliplr(img[0:length, 0:cropSize])
	mirror_image[cropSize:cropSize + length, -cropSize:] = np.fliplr(img[0:length, -cropSize:])
	mirror_image[-cropSize:, cropSize:cropSize + length] = np.flipud(img[-cropSize:, 0:length])
	return mirror_image

def shuffle_together(data_set):
	xlist = range(data_set[0].shape[0])
	xlist = random.sample(xlist, len(xlist))
	new_data_set = []
	for module in range(len(data_set)):
		newdata = data_set[module].copy()
		for dataIndex in range(len(xlist)):
			newdata[dataIndex] = data_set[module][xlist[dataIndex]]

		new_data_set.append(newdata)

	return new_data_set

def generate_dataset(dataset = None, cropSize = 0, csZ = 0, split = None, seg = None, doDilation=False, doEdt=False):

	if dataset == "ecs_train":
		filename = ecs_train_filename
	elif dataset == "ecs_test":
		filename = ecs_test_filename
	elif dataset == "ecs_train_halfres":
		filename = ecs_train_halfres_filename
	elif dataset == "ecs_test_reann":
		filename = ecs_test_reann_filename
	elif dataset == "ecs_test_lee":
		filename = ecs_test_lee_filename
	elif dataset == "ac3":
		filename = ac3_filename
	elif dataset == "ac4":
		filename = ac4_filename
	else:
		print "Missing dataset"

	(grayImages, synapseImages) = np.load(filename)

	if seg == "short":
		grayImages = grayImages[0:15]
		synapseImages = synapseImages[0:15]

	img_shape = grayImages[0].shape
	grayImages_mirrored = np.zeros((grayImages.shape[0] + 2*csZ,
									img_shape[0] + 2*cropSize,
									img_shape[1] + 2*cropSize))

	for ind in range(np.shape(grayImages)[0]):
		img = (grayImages[ind] * 1.0 / 255).copy()
		grayImages[ind] = img
		grayImages_mirrored[ind+csZ, :, :] = mirror_image_layer(img, cropSize)

		if doDilation:
			synapseImages[ind] = morph.binary_dilation(synapseImages[ind], iterations=2)

	if csZ > 0:
		for i in range(csZ):
			grayImages_mirrored[i,:,:]=grayImages_mirrored[2*csZ-i,:,:]
		grayImages_mirrored[-(i+1),:,:]=grayImages_mirrored[-(2*csZ+1)+i,:,:]

	return (grayImages, grayImages_mirrored, synapseImages)

def generate_samples(nsamples = 1, patchSize = 320, patchSize_out = 0, patchZ = 8,
					 patchZ_out = 0, modelType = 0, doAugmentation = False, doBinary = False, dataset = None):
	while True:
		grayImages, grayImages_mirrored, synapseImages = dataset

		if patchSize_out == 0:
			patchSize_out = patchSize
		if patchZ_out == 0:
			patchZ_out = patchZ

		cropSize = (patchSize - patchSize_out) / 2
		csZ = (patchZ - patchZ_out) / 2
		grayImg_set = np.zeros((nsamples, patchZ, patchSize, patchSize))
		synapse_set = np.zeros(nsamples) if doBinary \
					  else np.zeros((nsamples, patchZ_out, patchSize_out, patchSize_out))

		num = 0
		while num < nsamples:
			x_index = random.randint(0, 2 * cropSize + grayImages.shape[2] - patchSize)
			y_index = random.randint(0, 2 * cropSize + grayImages.shape[1] - patchSize)
			z_index = random.randint(0, 2 * csZ + grayImages.shape[0] - patchZ)

			grayImg_set[num, :, :, :] = grayImages_mirrored[z_index:z_index + patchZ,
															y_index:y_index + patchSize,
															x_index:x_index + patchSize]

			synapsePatch = synapseImages[z_index:z_index + patchZ_out,
										 y_index:y_index + patchSize_out,
										 x_index:x_index + patchSize_out]

			if doBinary:

				synapse_set[num] = float(len(np.unique(synapsePatch))>1)
				num += 1

			else:

				synapse_set[num] = synapsePatch

				if np.unique(synapse_set[num]).shape[0] < 2:
					continue

				if doAugmentation:

					if random.randint(0, 1):
						grayImg_set[num, :, :, :] = grayImg_set[num, ::-1, :, :]
						synapse_set[num, :, :, :] = synapse_set[num, ::-1, :, :]
					if random.randint(0, 1):
						grayImg_set[num, :, :, :] = grayImg_set[num, :, ::-1, :]
						synapse_set[num, :, :, :] = synapse_set[num, :, ::-1, :]
					if random.randint(0, 1):
						grayImg_set[num, :, :, :] = grayImg_set[num, :, :, ::-1]
						synapse_set[num, :, :, :] = synapse_set[num, :, :, ::-1]
					if random.randint(0, 1):
						grayImg_set[num, :, :, :] = grayImg_set[num].transpose((0, 2, 1))
						synapse_set[num, :, :, :] = synapse_set[num].transpose((0, 2, 1))

				num += 1

		if "runet" in modelType: # recurrent
			newSynapse = np.zeros((nsamples, patchZ_out, patchSize_out ** 2))
			for i in range(nsamples):
				newSynapse[i] = synapse_set[i].reshape(patchZ_out, patchSize_out ** 2)
			grayImg_set = np.reshape(grayImg_set.astype(np.float32), [-1, patchZ, 1, patchSize, patchSize])
		elif modelType == "unet_2d": # 2d
			newSynapse = np.zeros((nsamples, patchSize_out ** 2))
			for i in range(nsamples):
				newSynapse[i] = synapse_set[i][0].flatten()
			grayImg_set = np.reshape(grayImg_set.astype(np.float32)[:, 0, :, :], [-1, 1, patchSize, patchSize])
		elif "unet_25d" in modelType or "unet_3d" in modelType: #3d or 2.5d
			newSynapse = np.zeros((nsamples, patchZ_out*patchSize_out ** 2))
			for i in range(nsamples):
				newSynapse[i] = synapse_set[i].flatten()
			grayImg_set = np.reshape(grayImg_set.astype(np.float32), [-1, 1, patchZ, patchSize, patchSize])
		elif "binary" in modelType:
			newSynapse = synapse_set
			grayImg_set = np.reshape(grayImg_set.astype(np.float32)[:, 0, :, :], [-1, 1, patchSize, patchSize])
			if modelType == "binary_vgg16": grayImg_set = np.repeat(grayImg_set, 3, axis=1)
		newSynapse = newSynapse.astype(np.float32)
		data_set = (grayImg_set, newSynapse)
		yield data_set


def validate_model(model=0, patchSize=320, patchSize_out=0, patchZ=4, patchZ_out=0, modelType=0, dataset=None, bufferSize=0, save_predictions=False):
	grayImages, grayImages_mirrored, synapseImages = dataset

	if patchSize_out == 0:
		patchSize_out = patchSize
	if patchZ_out == 0:
		patchZ_out = patchZ

	cropSize = (patchSize - patchSize_out) / 2
	csZ = (patchZ - patchZ_out) / 2

	numImages = grayImages.shape[0]
	img = grayImages[0]
	probImages = np.zeros(grayImages.shape)
	numSample_axis = int(grayImages.shape[1] / patchSize_out) + 1
	numSample_patch = numSample_axis ** 2
	zSize = patchZ_out-bufferSize
	numZ = int(numImages / zSize) + 1
	nsamples = numZ * numSample_patch

	print 'Total number of validation samples:', nsamples
	grayImg_set = np.zeros((nsamples, patchZ, patchSize, patchSize))
	synapse_set = np.zeros((nsamples, patchZ_out, patchSize_out, patchSize_out))

	num_total = 0
	for zIndex in range(numZ-1):
		if zIndex == numZ - 2:
			zStart = numImages - patchZ_out
		else:
			zStart = zSize * zIndex

		for xIndex in range(numSample_axis - 1):
			xStart = patchSize_out * xIndex
			for yIndex in range(numSample_axis - 1):
				yStart = patchSize_out * yIndex
				grayImg_set[num_total] = grayImages_mirrored[zStart:zStart + patchZ, xStart:xStart + patchSize, yStart:yStart + patchSize]
				num_total += 1

		xStart = img.shape[0] - patchSize_out
		for yIndex in range(numSample_axis - 1):
			yStart = patchSize_out * yIndex
			grayImg_set[num_total] = grayImages_mirrored[zStart:zStart + patchZ, xStart:xStart + patchSize, yStart:yStart + patchSize]
			num_total += 1

		yStart = img.shape[1] - patchSize_out
		for xIndex in range(numSample_axis - 1):
			xStart = patchSize_out * xIndex
			grayImg_set[num_total] = grayImages_mirrored[zStart:zStart + patchZ, xStart:xStart + patchSize, yStart:yStart + patchSize]
			num_total += 1

		xStart = img.shape[0] - patchSize_out
		yStart = img.shape[1] - patchSize_out
		grayImg_set[num_total] = grayImages_mirrored[zStart:zStart + patchZ, xStart:xStart + patchSize, yStart:yStart + patchSize]
		num_total += 1

	print 'Load in the model'
	prev_time = int(time.time())
	print "Start time", prev_time
	for val_ind in range(nsamples):
		if val_ind % 10 == 0:
			print 'Validating sample #', val_ind, 'out of', grayImg_set.shape[0]
		data_x = grayImg_set[val_ind].astype(np.float32)
		if "runet" in modelType: # recurrent
			data_x = np.reshape(data_x, [-1, patchZ, 1, patchSize, patchSize])
		elif modelType == "unet_2d":
			data_x = np.reshape(data_x, [-1, 1, patchSize, patchSize])
		elif "unet_25d" in modelType or "unet_3d" in modelType:
			data_x = np.reshape(data_x, [-1, 1, patchZ, patchSize, patchSize])
		im_pred = model.predict(x=data_x, batch_size=1)
		#im_pred = normalize(im_pred)
		synapse_set[val_ind] = np.reshape(im_pred, (patchZ_out, patchSize_out, patchSize_out))
	print "Time elapsed", int(time.time())-prev_time
	print 'Write images'
	num_total = 0
	for zIndex in range(numZ-1):
		if zIndex == numZ - 2:
			zStart = numImages - patchZ_out
		else:
			zStart = zSize * zIndex
		for xIndex in range(numSample_axis - 1):
			xStart = patchSize_out * xIndex
			for yIndex in range(numSample_axis - 1):
				yStart = patchSize_out * yIndex
				probImages[zStart + bufferSize:zStart + patchZ_out, xStart:xStart + patchSize_out, yStart:yStart + patchSize_out] = synapse_set[num_total, bufferSize:, :, :]
				num_total += 1

		xStart = (numSample_axis - 1) * patchSize_out
		for yIndex in range(numSample_axis - 1):
			yStart = patchSize_out * yIndex
			probImages[zStart + bufferSize:zStart + patchZ_out, xStart:, yStart:yStart + patchSize_out] = synapse_set[num_total, bufferSize:, xStart - img.shape[0]:, :]
			num_total += 1

		yStart = (numSample_axis - 1) * patchSize_out
		for xIndex in range(numSample_axis - 1):
			xStart = patchSize_out * xIndex
			probImages[zStart + bufferSize:zStart + patchZ_out, xStart:xStart + patchSize_out, yStart:] = synapse_set[num_total, bufferSize:, :, yStart - img.shape[0]:]
			num_total += 1

		xStart = (numSample_axis - 1) * patchSize_out
		yStart = (numSample_axis - 1) * patchSize_out
		probImages[zStart + bufferSize:zStart + patchZ_out, xStart:, yStart:] = synapse_set[num_total, bufferSize:, xStart - img.shape[0]:, yStart - img.shape[0]:]
		num_total += 1

	if save_predictions:
		np.save('/n/coxfs01/eric_wu/synapse/predictions/' + save_predictions + '.npy', probImages)
	return (probImages, synapseImages)