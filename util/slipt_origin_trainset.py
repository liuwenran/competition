import cv2
import os
import shutil
import numpy as np

pic_train_path = '/home/liuwr/liuwenran/competition/officialData/dataset/formalCompetition4/pic_resize_train'
pic_validate_path = '/home/liuwr/liuwenran/competition/officialData/dataset/formalCompetition4/pic_resize_val'

vec_train_path = '/home/liuwr/liuwenran/competition/officialData/dataset/formalCompetition4/docFeature_train'
vec_val_path = '/home/liuwr/liuwenran/competition/officialData/dataset/formalCompetition4/docFeature_val'

all_pic_list = os.listdir(pic_train_path)

if len(all_pic_list) > 91000:
	val_file_list = all_pic_list[-10000:]

	for i in range(len(val_file_list)):
		filename = val_file_list[i]
		origin_file = os.path.join(pic_train_path, filename )
		dest_file = os.path.join(pic_validate_path, filename)
		shutil.move(origin_file, dest_file)

		dotind = filename.rfind('.')
		namebdot = filename[:dotind]
		origin_vec_file = os.path.join(vec_train_path, namebdot + '.npy')
		dest_vec_file = os.path.join(vec_val_path, namebdot + '.npy')
		shutil.move(origin_vec_file, dest_vec_file)




