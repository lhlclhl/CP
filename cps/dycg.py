''' dynamic candidate generation: generate candidates by the letters on the grid
	1. MultiWordGenerator: generate phrases
	2. LetterSequenceGenerator: generate letter sequences

TODO:
1. pair-wise consistency: P(w1) * max(P(w2), P(w2|w1)) * ...
'''
import re, string, math, sys, numpy as np, os, json
from os.path import join, exists
from collections import defaultdict

from keras.layers import *
from keras.models import *
from keras.callbacks import *
import keras.backend as K

from .puz_utils import string2fill

class MultiWordGenerator:
	@property
	def model_path(self):
		return join("models/multiword", "params.json")
	def __init__(self, wfreq_path="data/dictionaries/unigram_freq_filtered.csv", \
	vocabfile="data/dictionaries/vocab.txt", minFreq=100):
		self.wd2id = {}
		self.id2wd = []
		self.wpriors = []
		self.total = 0
		with open(wfreq_path, encoding="utf-8") as fin:
			fin.readline()
			for line in fin:
				wd, ouni = line.strip("\r\n").rsplit(",", 1)
				ouni = float(ouni)
				if ouni >= minFreq:# and wd in wset:
					self.total += ouni
					wd = string2fill(wd)#re.sub("[%s]+"%string.punctuation, "", wd).strip()
					if wd not in self.wd2id:
						self.wd2id[wd] = len(self.id2wd)
						self.id2wd.append(wd)
					wid = self.wd2id[wd]
					if wid >= len(self.wpriors): self.wpriors.append(0)
					self.wpriors[wid] += ouni
		print(len(self.wpriors), "words")
		for i in range(len(self.wpriors)):
			self.wpriors[i] = math.log(self.wpriors[i])

		self._strings = []
		for i, s in enumerate(self.id2wd):
			while len(s) >= len(self._strings):
				self._strings.append(defaultdict(set))
			for j, c in enumerate(s):
				self._strings[len(s)][(j, c)].add(i)
			self._strings[len(s)].setdefault(0, []).append(i)
		
		try: self.load(self.model_path)
		except Exception: 
			self.adapt(self.model_path, vocabfile)
		# smoothing
		for i in range(1, len(self.upper_lower_by_length)-1):
			lower, upper = self.upper_lower_by_length[i]
			if lower == 0.: 
				lower = (self.upper_lower_by_length[i-1][0]+self.upper_lower_by_length[i+1][0])/2
			if upper == 0.:
				upper = (self.upper_lower_by_length[i-1][1]+self.upper_lower_by_length[i+1][1])/2
			self.upper_lower_by_length[i] = (lower, upper)
	def adapt(self, mod_file, vocabfile, n_samples=1000):
		words_by_lengths = []
		charcnt = defaultdict(int)
		with open(vocabfile, encoding="utf-8") as fin:
			for line in fin:
				word, srcs = line.strip("\r\n").split("\t")
				if srcs.startswith("1"):
					while len(word) >= len(words_by_lengths):
						words_by_lengths.append([])
					words_by_lengths[len(word)].append(word)
					for c in word: charcnt[c] += 1
		chars, cdist = zip(*charcnt.items())
		chars = np.array(chars)
		cdist = np.array(cdist, dtype="float32")
		cdist /= cdist.sum()

		self.upper_lower_by_length = []
		for i in range(len(words_by_lengths)):
			upper = lower = 0
			if words_by_lengths[i]:
				for j in np.random.choice(len(words_by_lengths[i]), n_samples):
					word = words_by_lengths[i][j]
					_, s = self._generate(word)
					upper += s
				for cs in np.random.choice(len(cdist), (n_samples, i), p=cdist):
					word = "".join(chars[cs])
					_, s = self._generate(word)
					lower += s
			self.upper_lower_by_length.append((lower/n_samples, upper/n_samples))
		self.save(mod_file)
	def save(self, mod_file):
		with open(mod_file, "w", encoding="utf-8") as fout: 
			json.dump(self.upper_lower_by_length, fout, ensure_ascii=False, indent=4)
	def load(self, mod_file): 
		self.upper_lower_by_length = json.load(open(mod_file, encoding="utf-8"))
	def filter(self, pattern):
		if "*" not in pattern: 
			if pattern in self.wd2id: return [self.wd2id[pattern]]
			else: return []
		cand = []
		for t, c in enumerate(pattern):
			if c != "*":
				cs = self._strings[len(pattern)][(t, c)]
				cand.append(cs)
		if cand: 
			cand = list(set.intersection(*cand))
		elif len(pattern) == 1: cand = self._strings[1][0]
		return cand
	def select(self, cands):
		bw, ls = None, 0
		if cands:
			for w in cands:
				s = self.wpriors[w]
				if s > ls:
					bw, ls = w, s
		return bw, ls
	def _generate(self, cseq, dec=1., blacklist=set()):
		N = len(cseq)
		DAG = []
		for i in range(N):
			cands = {}
			for j in range(i, N):
				r = self.filter(cseq[i:j+1].upper())
				bw, ls = self.select(r)
				if bw is not None: cands[j] = (bw, ls)
			if not cands: 
				print("dycg: bad cseq", cseq)
				return cseq, -1e9
			DAG.append(cands)
		#print(DAG)
		route = {N:(0, 0)}		
		logtotal = math.log(self.total)*dec
		for idx in range(N-1, -1, -1):
			route[idx] = max((s-logtotal + route[j + 1][0], j) for j, (w, s) in DAG[idx].items())
		#print("best route", route[0])
		x = 0
		ret = []
		while x < N:
			y = route[x][1]
			ret.append(self.id2wd[DAG[x][y][0]])
			x = y + 1
		return ret, route[0][0]
	def generate(self, cseq, dec=1., blacklist=set()):
		ret, score = self._generate(cseq, dec, blacklist)
		lower, upper = self.upper_lower_by_length[len(cseq)]
		rs = max(min((score-lower)/(upper-lower), 1), 0)
		return ret, rs
class MWG(MultiWordGenerator):
	@property
	def model_path(self):
		return join("models/multiword", "params1.json")
	def build_model(self):
		inp = Input((1,), dtype="float32")
		outs = Dense(1, activation="sigmoid")(inp)
		self.model = Model(inp, outs)
		self.model.compile("adam", "binary_crossentropy", ["accuracy"])
	def adapt(self, mod_file, vocabfile, n_samples_base=1000, posr=0.1):
		mod_dir = os.path.split(mod_file)[0]
		words_by_lengths = []
		charcnt = defaultdict(int)
		with open(vocabfile, encoding="utf-8") as fin:
			for line in fin:
				word, srcs = line.strip("\r\n").split("\t")
				if srcs.startswith("1"):
					while len(word) >= len(words_by_lengths):
						words_by_lengths.append([])
					words_by_lengths[len(word)].append(word)
					for c in word: charcnt[c] += 1
		chars, cdist = zip(*charcnt.items())
		chars = np.array(chars)
		cdist = np.array(cdist, dtype="float32")
		cdist /= cdist.sum()

		self.build_model()
		self.upper_lower_by_length = []
		for i in range(len(words_by_lengths)):
			n_samples = n_samples_base * i
			W, b = 0., 0.
			if words_by_lengths[i] and n_samples:
				pos = []
				phs = []
				for j in np.random.choice(len(words_by_lengths[i]), int(n_samples*posr)):
					word = words_by_lengths[i][j]
					ws, s = self._generate(word)
					pos.append(s)
					if len(ws) > 1:
						phs.append("".join(ws))
				neg = []
				if phs:
					n = min(i*len(phs), int(n_samples*(1-posr)/2))
					inds = np.random.choice(len(phs), n)
					ps = np.random.choice(i, n)
					chs = np.random.choice(len(chars), n, p=cdist)
					for j in range(n):
						w = phs[inds[j]][:ps[j]] + chars[chs[j]] + phs[inds[j]][ps[j]+1:]
						_, s = self._generate(w)
						neg.append(s)
				for cs in np.random.choice(len(cdist), (n_samples-len(pos)-len(neg), i), p=cdist):
					word = "".join(chars[cs])
					_, s = self._generate(word)
					neg.append(s)
				
				X = np.array(pos + neg)
				Y = np.array([1.] * len(pos) + [0.] * len(neg))
				val_size = len(X)//10
				np.random.seed(1333); np.random.shuffle(X)
				np.random.seed(1333); np.random.shuffle(Y)
				valX, traX = X[:val_size], X[val_size:]
				valY, traY = Y[:val_size], Y[val_size:]

				session = K.get_session()
				for layer in self.model.layers: 
					if hasattr(layer, 'kernel_initializer'):
						layer.kernel.initializer.run(session=session)

				mod_file = join(mod_dir, "%d.h5"%i)
				self.model.fit(traX, traY, epochs=50, validation_data=(valX, valY), callbacks=[
					ModelCheckpoint(mod_file, save_best_only=True, save_weights_only=True)
				])
				self.model.load_weights(mod_file)
				W, b = self.model.get_weights()
			self.upper_lower_by_length.append((float(W), float(b)))
		self.save(self.model_path)
	def generate(self, cseq, dec=1., blacklist=set()):
		ret, score = self._generate(cseq, dec, blacklist)
		W, b = self.upper_lower_by_length[len(cseq)]
		return ret, 1/(1+np.exp(-W*score-b))
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
from .base import BaseModel, Tokenizer
from keras.layers import *
from keras.models import *
from keras.callbacks import *
class LetterSequenceGenerator(BaseModel):
	@property
	def fields_to_save(self): return dict(super().fields_to_save, **{
		"emb_dims" : 256,
		"lstm_dims": 128,
		"dropout": 0.2,
		"n_layers": 1,
		"max_mask_ratio": 0.5,
		"max_len": 15,
		"batch_size": 128,
	})
	@property
	def options(self): return dict(super().fields_to_save, **{
		"model_dir": "./models/lseqgen",
	})
	def build_tokenizers(self, data=None):
		samples, labels =(None, None) if data is None else data
		self._tokenizers = [Tokenizer(samples, special_tokens={"mask":"*"})]
		self._classes = []
		self._data_generator = lambda x,y:(x,y) # no generator by default
	def build_model(self):
		inpseq = Input((None,), dtype="int32")
		x = Embedding(len(self._tokenizers[0]), self.emb_dims, mask_zero=True)(inpseq)
		for _ in range(self.n_layers):
			x_fwd = LSTM(self.lstm_dims, return_sequences=True, dropout=self.dropout)(x)
			x_bwd = LSTM(self.lstm_dims, return_sequences=True, dropout=self.dropout, go_backwards=True)(x)
			x = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=-1))([x_fwd, x_bwd])
		outps = Dense(len(self._tokenizers[0]), activation="softmax")(x)
		self._model = self._pmodel = Model(inpseq, outps)
		self._model.compile("Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
		self._model.summary()
	def load_data(self, datafile, verbose=False, limit=0):	
		samples = []
		labels = []
		t = 0
		print("loading data")
		with open(datafile, encoding="utf-8") as fin:
			for line in fin:
				st = line.strip("\r\n")
				if len(st) <= self.max_len and re.fullmatch("[A-Z]+", st):
					samples.append(st)
					labels.append(0)
				t += 1
		print("samples loaded %d/%d=%.3f"%(len(samples), t, len(samples)/t))
		return samples[-limit:], labels[-limit:]
	def make_training_data(self, data):
		samples, labels = data
		ss = []
		tt = []
		for st in samples:
			max_mask_num = int(len(st)*self.max_mask_ratio)
			for i in range(len(st)-2):
				tt.append(st)
				s = list(st)
				for j in np.random.choice(len(st), np.random.randint(0, max_mask_num)+1):
					s[j] = "*"
				ss.append(s)
		
		X = self.make_feature(ss)
		Y = [yy[:,:,None] for yy in self.make_feature(tt)]
		print("X", [xx.shape for xx in X])
		print("Y", [yy.shape for yy in Y])
		return X, Y
	def do_batch(self, samples):
		feats = self.make_feature(samples)
		probs = self.predict(feats)[0]
		preds = probs.argmax(axis=2)
		ret = []
		for seq in preds:
			ret.append("".join(self._tokenizers[0].token(c) for c in seq if c > 0))
		return ret

if __name__ == "__main__":
	mwg = MWG()#MultiWordGenerator()
	print(mwg.generate("FI*TYCENTPEAS"))
	print(mwg.generate("FREENOWNLOAD"))
	print(mwg.generate("FREE*OWNLOAD"))
	print(mwg.generate("THEMU*FINMANET"))
	print(mwg.generate("*EF*TOWN"))
	print(mwg.generate("TENTTOWN"))
	print(mwg.generate("FI*DAGAS*T**I*N"))
	print(mwg.generate("ARC*ICCHARITY"))
	print(mwg.generate("THE"))
	print(mwg.generate("GOOEK"))
	print(mwg.generate("KJHG"))
	# lsg = LetterSequenceGenerator(sys.argv)
	# #lsg.train("intermediate/clues_index/fill_strings.txt")
	# lsg.load()
	# print(lsg.do("ST*TI*N"))
