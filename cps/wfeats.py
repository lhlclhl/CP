import h5py, numpy as numpy, math
from os.path import join
from .puz_utils import string2fill

w2v_path = "data/w2v" # word2vec directory
pfeat_path = "data/dictionaries/stringfeats.txt" # word prior feature file
wfreq_path="data/blanks/unigram_freq.csv"
idf_path = "data/idf.txt"

wd2id, wvecs = None, None
def get_w2v(w2v_path=w2v_path):
	global wd2id, wvecs
	if wvecs is None or wd2id is None:
		words = [line.strip("\r\n")for line in open(join(w2v_path, "words.txt"), encoding="utf-8")]
		wd2id = {y:x for x, y in enumerate(words)}
		with h5py.File(join(w2v_path, "vectors.h5"), "r") as dfile:
			wvecs = dfile["data"][:]
		print("w2v loaded")
	return wd2id, wvecs

vocabulary = None
def get_vocab(path):
	global vocabulary
	if vocabulary is None:
		vocabulary = set()
		with open(path, encoding="utf-8") as fin:
			for line in fin:
				word, code = line.strip("\r\n").split("\t")
				if int(code) != 2: # not generated from wikititles
					vocabulary.add(word)
	return vocabulary

idf, max_idf = None, None
def get_idf(path=idf_path):
	global idf, max_idf
	if idf is None:
		idf, max_idf = {}, 0
		with open(path, encoding="utf-8") as fin:
			for line in fin:
				word, f = line.strip().split("\t")
				f = float(f)
				if f > max_idf: max_idf = f
				idf[word] = f
	return idf, max_idf

wfreqs = None
def get_wprior(pfeat_path=pfeat_path, wfreq_path=wfreq_path):
	global wfreqs
	if wfreqs is None:
		print("loading word prior", pfeat_path)
		wfreqs = {}
		with open(pfeat_path, encoding="utf-8") as fin:
			for line in fin:
				wd, odb, ocwg, ouni, wt = line.strip("\r\n").split("\t")

				fwd = string2fill(wd)
				if fwd not in wfreqs: wfreqs[fwd] = [0, 0, 0, 0]
				wfreqs[fwd][0] += int(odb)
				wfreqs[fwd][1] += int(ocwg)
				wfreqs[fwd][3] += int(wt)
		print("loading word freq", wfreq_path)
		with open(wfreq_path, encoding="utf-8") as fin:
			fin.readline()
			for line in fin:
				wd, ouni = line.strip("\r\n").rsplit(",", 1)
				fwd = string2fill(wd)
				ouni = int(ouni)
				if fwd not in wfreqs: wfreqs[fwd] = [0, 0, 0, 0]
				wfreqs[fwd][2] += ouni
		for k in wfreqs:
			for i in range(4):
				wfreqs[k][i] = math.log(1+wfreqs[k][i])
		print("word prior loaded")
	return wfreqs

if __name__ == "__main__":
	get_wprior()
#wpriors = {}
# with open(pfeat_path, encoding="utf-8") as fin:
# 	for line in fin:
# 		wd, odb, ocwg, ouni, wt = line.strip("\r\n").split("\t")
# 		#wpriors[wd] = [math.log(int(odb)+1), math.log(int(ocwg)+1), math.log(int(ouni)+1), int(wt)]

# 		fwd = string2fill(wd)
# 		if fwd not in wfreqs: wfreqs[fwd] = [0, 0, 0, 0]
# 		wfreqs[fwd][0] += int(odb)
# 		wfreqs[fwd][1] += int(ocwg)
# 		wfreqs[fwd][3] += int(wt)
# with open(wfreq_path, encoding="utf-8") as fin:
# 	fin.readline()
# 	for line in fin:
# 		wd, ouni = line.strip("\r\n").rsplit(",", 1)
# 		fwd = string2fill(wd)
# 		ouni = int(ouni)
# 		if fwd not in wfreqs: wfreqs[fwd] = [0, 0, 0, 0]
# 		wfreqs[fwd][2] += ouni
# for k in wfreqs:
# 	for i in range(4):
# 		wfreqs[k][i] = math.log(1+wfreqs[k][i])
