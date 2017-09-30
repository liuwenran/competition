from gensim.models import word2vec
#from gensim.models.word2vec import LineSentence
import os

filepath = '/home/liuwr/liuwenran/competition/officialData/dataset/formalCompetition4/News_info_train'
cutfilepath = '/home/liuwr/liuwenran/competition/officialData/dataset/formalCompetition4/News_info_train_cut'
all_text_file = '/home/liuwr/liuwenran/competition/officialData/dataset/formalCompetition4/all_train_text.txt'

filelist = os.listdir(cutfilepath)
firstfile = os.path.join(cutfilepath, filelist[0]);
print firstfile
# firstbin = os.path.join('/home/liuwr/liuwenran/competition/thulac/THULAC-Python', 'firstfile.bin')

# word2vec.word2vec(firstfile, firstbin, size = 200, verbose = True)
sentences = word2vec.LineSentence(all_text_file)
model = word2vec.Word2Vec( min_count = 1, size=200)
model.build_vocab(sentences)
model.train(sentences, total_examples = model.corpus_count, epochs = model.iter)
model.save('all_text_model')
# model.build_vocab(firstfile)
# model.train(firstfile)
# model = word2vec.load(firstbin)
# model.vocab


