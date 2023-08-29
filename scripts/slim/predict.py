import tensorflow as tf
slim = tf.contrib.slim
import sys
import os
#import matplotlib.pyplot as plt
import numpy as np
import os
from nets import inception
from preprocessing import inception_preprocessing
from os import listdir
from os.path import isfile, join
from os import walk
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, precision_recall_fscore_support
import matplotlib.pyplot as plt

session = tf.Session()

def get_test_images(mypath):

	"""
	Gets a list of all JPEG image files in a test path.

	Args:
	path: Path to test image directory

	Returns:
	List of image file paths
	"""
	
	return [mypath + '/' + f for f in listdir(mypath) if isfile(join(mypath, f)) and f.find('.jpg') != -1]

def transform_img_fn(path_list):
    out = []
    for f in path_list:
        image_raw = tf.image.decode_jpeg(open(f,'rb').read(), channels=3)
        image = inception_preprocessing.preprocess_image(image_raw, image_size, image_size, is_training=False)
        out.append(image)
    return session.run([out])[0]


if __name__ == '__main__':
	

	if len(sys.argv) != 6:
		print("The script needs five arguments.")
		print("The first argument should be the CNN architecture: v1, v3 or inception_resnet2")
		print("The second argument should be the directory of trained model.")
		print("The third argument should be directory of test images.")
		print("The  fourth argument should be output file for predictions.")
		print("The  fifth argument should be number of classes.")
		exit()
	deep_lerning_architecture = sys.argv[1]
	train_dir = sys.argv[2]
	test_path = sys.argv[3]
	output = sys.argv[4]
	nb_classes = int(sys.argv[5])

	if deep_lerning_architecture == "v1" or deep_lerning_architecture == "V1":
		image_size = 224
	else:
		if deep_lerning_architecture == "v3" or deep_lerning_architecture == "V3" or deep_lerning_architecture == "resv2" or deep_lerning_architecture == "inception_resnet2":
			image_size = 299
		else:
			print("The selected architecture is not correct.")
			exit()


	print('Start to read images!')
	image_list = get_test_images(test_path)
	processed_images = tf.placeholder(tf.float32, shape=(None, image_size, image_size, 3))

	if deep_lerning_architecture == "v1" or deep_lerning_architecture == "V1":
		with slim.arg_scope(inception.inception_v1_arg_scope()):
			logits, _ = inception.inception_v1(processed_images, num_classes=nb_classes, is_training=False)

	else:
		if deep_lerning_architecture == "v3" or deep_lerning_architecture == "V3":
			with slim.arg_scope(inception.inception_v3_arg_scope()):
				logits, _ = inception.inception_v3(processed_images, num_classes=nb_classes, is_training=False)
		else:
			if deep_lerning_architecture == "resv2" or deep_lerning_architecture == "inception_resnet2":
				with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
					logits, _ = inception.inception_resnet_v2(processed_images, num_classes=nb_classes, is_training=False)

	def predict_fn(images):
	    return session.run(probabilities, feed_dict={processed_images: images})

	probabilities = tf.nn.softmax(logits)
	checkpoint_path = tf.train.latest_checkpoint(train_dir)
	init_fn = slim.assign_from_checkpoint_fn(checkpoint_path,slim.get_variables_to_restore())
	init_fn(session)
	label_dict = {0:'Good', 1:'Poor'}
	print('Start to transform images!')
	images = transform_img_fn(image_list)
	print(image_list)
	fto = open(output, 'w')
	fto.write('File name \t Ground truth \t Predicted label \n')
	print('Start doing predictions!')
	preds = predict_fn(images)
	print (len(preds))
	y_actual_list = list()
	y_pred_list = list()
	for p in range(len(preds)):
		pred = np.argmax(preds[p,:])
		print (image_list[p], preds[p,:], pred)
		img_name = os.path.basename(image_list[p])
		if img_name.startswith('good'):
			y_actual = 0
			y_actual_list.append(y_actual)
		else:
			y_actual = 1
			y_actual_list.append(y_actual)
		y_pred_list.append(pred)
		fto.write(image_list[p])
		fto.write('\t' + label_dict[y_actual] + '\t' + label_dict[pred])
		fto.write('\n')

	fto.close()

	# Display Confusion matrix
	conf_matrix = confusion_matrix(y_actual_list, y_pred_list)
	disp = ConfusionMatrixDisplay(conf_matrix, display_labels=['Good', 'Poor'])
	disp.plot()
	plt.show()

	# Evaluation metrics
	precision, recall, f1_score, support = precision_recall_fscore_support(y_actual_list, y_pred_list, average='weighted')
	print(f'Specificity:{precision}\nSensitivity:{recall}\nF1-score:{f1_score}')
	print('Accuracy:' + str(accuracy_score(y_actual_list, y_pred_list)))

