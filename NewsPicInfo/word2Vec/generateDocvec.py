from gensim.models import word2vec
#from gensim.models.word2vec import LineSentence
import os
import numpy as np
import thulac

filepath = '/home/liuwr/liuwenran/competition/officialData/dataset/formalCompetition4/News_info_train'
cutfilepath = '/home/liuwr/liuwenran/competition/officialData/dataset/formalCompetition4/News_info_train_cut'
all_text_file = '/home/liuwr/liuwenran/competition/officialData/dataset/formalCompetition4/all_train_text.txt'
featurePath = '/home/liuwr/liuwenran/competition/officialData/dataset/formalCompetition4/News_info_train_docFeature/'

filelist = os.listdir(cutfilepath)


featureWidth = 20;
featureLen = 200;
# print firstfile
# firstbin = os.path.join('/home/liuwr/liuwenran/competition/thulac/THULAC-Python', 'firstfile.bin')

allmodel = word2vec.Word2Vec.load('all_text_model')
# allkeys = allmodel.wv.vocab.keys()
thu1 = thulac.thulac()

# word2vec.word2vec(firstfile, firstbin, size = 200, verbose = True)
#for fileind in range(len(filelist))
for fileind in range(97500,100000):
	print str(fileind) + ' in ' + str(len(filelist))
	firstfile = os.path.join(cutfilepath, filelist[fileind]); 
	sentences = word2vec.LineSentence(firstfile)
	curmodel = word2vec.Word2Vec( min_count = 1, size=200)
	curmodel.build_vocab(sentences)

	wordlist = []
	countlist = []
	for word, vocab_obj in curmodel.wv.vocab.items():
		wordlist.append(word)
		countlist.append(vocab_obj.count)

	sortind = np.argsort(-np.array(countlist))
	sort_word_list = [wordlist[i] for i in sortind]
	sort_count_list = sorted(countlist, reverse = True)

	# for i in range(20):
	# 	print sort_word_list[i],sort_count_list[i]
	excludeSet = ['w', 'r', 'c', 'p', 'u', 'y']
	top20word = []
	i = 0
	count = 0
	while(count < featureWidth and i < len(sort_word_list)):
		newword = thu1.cut(sort_word_list[i].encode('utf-8'))
		if newword[0][1] not in excludeSet:
			top20word.append(sort_word_list[i])
			count = count + 1
		i = i + 1

	# print ''
	docFeature = np.zeros((featureLen, featureWidth))
	for i in range(count):
		# if top20word[i] in allkeys:
		docFeature[:,i] = allmodel.wv[top20word[i]]

	tempname = filelist[fileind]
	tempname = tempname[:-4]
	np.save(featurePath + tempname + '.npy', docFeature)





