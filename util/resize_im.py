import cv2
import os
import numpy as np

train_im_dir = '/home/liuwr/liuwenran/competition/officialData/dataset/formalCompetition4/News_pic_info_train'
output_im_dir = '/home/liuwr/liuwenran/competition/officialData/dataset/formalCompetition4/News_pic_info_train_resize'

filelist = os.listdir(train_im_dir)

notpic = []
notpic_count = 0

for i in range(len(filelist)):
	print 'im ' + str(i) + ' in ' + str(len(filelist))
	imfile = os.path.join(train_im_dir, filelist[i])
	im = cv2.imread(imfile)
	if type(im) == type(np.ndarray([1])) :
		outim = cv2.resize(im,(227,227))
		outimfile = os.path.join(output_im_dir, filelist[i])
		cv2.imwrite(outimfile,outim)
	else:
		notpic.append(filelist[i])
		notpic_count = notpic_count + 1

print 'notpic_count:' + str(notpic_count)
np.save('notpic_list.npy', notpic)