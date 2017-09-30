#coding:utf-8

import thulac
import os

thu1 = thulac.thulac(seg_only=True)  #设置模式为行分词模式

train_set_dir = '/home/liuwr/liuwenran/competition/officialData/dataset/formalCompetition4/News_info_train'
output_set_dir = '/home/liuwr/liuwenran/competition/officialData/dataset/formalCompetition4/News_info_train_cut'

filelist = os.listdir(train_set_dir)

for i in range(len(filelist)):
	print 'working on ' + str(i) + ' in '+ str(len(filelist))
	curfile = os.path.join(train_set_dir, filelist[i])
	cur_outfile = os.path.join(output_set_dir, filelist[i])
	thu1.cut_f(curfile, cur_outfile)
# a = thu1.cut("我爱北京天安门")

# print(a)
