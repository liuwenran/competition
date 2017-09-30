#coding:utf-8

import thulac
import os

thu1 = thulac.thulac(seg_only=True)  #设置模式为行分词模式

train_set_dir = '/home/liuwr/liuwenran/competition/officialData/dataset/formalCompetition4/News_info_train'
output_set_dir = '/home/liuwr/liuwenran/competition/officialData/dataset/formalCompetition4/News_info_train_cut'

filelist = os.listdir(train_set_dir)

shortlen = 5000

for i in range(len(filelist)):
	print 'working on ' + str(i) + ' in '+ str(len(filelist))
	curfile = os.path.join(train_set_dir, filelist[i])
	print curfile
	in_file_object = open(curfile)
	all_text = in_file_object.read()
	in_file_object.close()

	out_text = ''
	if(len(all_text) > shortlen):
		lastind = 0
		ind = len(all_text)
		flag = 0
		for j in range(int(len(all_text)/shortlen) ):
			if(j == int(len(all_text)/shortlen) - 1):
				ind = len(all_text)
			else:
				ind = all_text.find(' ', shortlen * (j+1))
			if(ind == -1):
				ind = len(all_text)
				flag = 1
			short_text = all_text[lastind:ind]
			lastind = ind
			out_text_temp = thu1.cut(short_text, text=True)
			out_text = out_text + out_text_temp
			if(flag == 1):
				break
	else:
		out_text = thu1.cut(all_text, text=True)

			
	cur_outfile = os.path.join(output_set_dir, filelist[i])
	# thu1.cut_f(curfile, cur_outfile)
	# out_text = all_text
	out_file_object = open(cur_outfile, 'w')
	out_file_object.write(out_text)
	out_file_object.close()

# print(a)
