#coding:utf-8

# import thulac
import os

# thu1 = thulac.thulac(seg_only=True)  #设置模式为行分词模式

# train_set_dir = '/home/liuwr/liuwenran/competition/officialData/dataset/formalCompetition4/News_info_train'
output_set_dir = '/home/liuwr/liuwenran/competition/officialData/dataset/formalCompetition4/News_info_train_cut'
all_text_file = '/home/liuwr/liuwenran/competition/officialData/dataset/formalCompetition4/all_train_text.txt'

filelist = os.listdir(output_set_dir)
out_file_object = open(all_text_file, 'w+')

for i in range(len(filelist)):
	print 'working on ' + str(i) + ' in '+ str(len(filelist))
	curfile = os.path.join(output_set_dir, filelist[i])
	print curfile
	in_file_object = open(curfile)
	all_text = in_file_object.read()
	in_file_object.close()

	# thu1.cut_f(curfile, cur_outfile)
	# out_text = all_text

	out_file_object.write(all_text)

out_file_object.close()

# print(a)
