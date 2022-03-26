import sys, argparse, json, string, os, random, traceback, inspect
from collections import defaultdict
from os.path import join, exists, dirname
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
from keras import backend as K
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.optimizers import *
from keras.losses import KLDivergence, categorical_crossentropy
from keras.metrics import accuracy
from .base import LogEvalSave, BaseModel
from keras.regularizers import l2


def crossentropy(y_true, y_pred):
	y_true = K.clip(y_true, K.epsilon(), 1)
	return -K.sum(y_true * K.log(y_pred), axis=-1)
def kullback_leibler_divergence(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.sum(y_true * K.log(y_true / y_pred), axis=-1)
def contrastive(margin=1.):
	def loss(y_true, y_pred):
		return K.sum((1-y_true) * K.abs(y_pred) + y_true * K.maximum(margin - y_pred, 0), axis=-1)
	def accuracy(y_true, y_pred):
		return K.cast((y_true * K.minimum(margin, y_pred) + (1-y_true) * (margin-y_pred))/margin, K.floatx())
	return loss, accuracy
def weighted_CE(epsilon=1e-7, temp=1, smooth=0.):
	def WCE(y_true, y_pred):
		y_pred = K.softmax(y_pred/temp, axis=-1)
		y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
		# scale preds so that the class probas of each sample sum to 1
		# y_pred /= K.sum(y_pred, axis=-1)
		# avoid numerical instability with epsilon clipping
		if smooth > 1e-7:
			y_pred = K.minimum(y_pred+smooth, 1.)
		else: y_pred = K.maximum(y_pred, epsilon)
		n_classes = K.int_shape(y_pred)[-1]
		y_true = K.cast(K.one_hot(K.squeeze(K.cast(y_true, "int32"), -1), n_classes), y_pred.dtype)

		y_target = K.sum(y_pred * y_true, -1, keepdims=True)
		tar_cmp = K.greater(y_pred, K.repeat_elements(y_target, n_classes, -1))
		tar_rank = K.sum(K.cast(tar_cmp, y_pred.dtype), axis=-1)
		weights = 1-1/(tar_rank+1) # weight = 1-MRR

		return -weights * K.sum(y_true * K.log(y_pred), axis=-1)
	return WCE
def MRR(y_true, y_pred):
	n_classes = K.int_shape(y_pred)[-1]
	y_true = K.cast(K.one_hot(K.squeeze(K.cast(y_true, "int32"), -1), n_classes), y_pred.dtype)
	y_target = K.sum(y_pred * y_true, -1, keepdims=True)
	tar_cmp = K.greater(y_pred, K.repeat_elements(y_target, n_classes, -1))
	tar_rank = K.sum(K.cast(tar_cmp, y_pred.dtype), axis=-1)
	return 1/(tar_rank+1)

class Rewarder(BaseModel):
	'''
	Global Rewarder: Make the Grid Reward distribution close to the metric distribution
	'''
	@property
	def params(self): return dict(super().params, **{
		"sample_size": 512,
		"loss": "CE",
		"n_features": 20,
	})
	@property
	def options(self): return dict(super().options, **{
		"model_dir": "./models/reward_new",
		"train_dir": "intermediate/reward_feats/MCTS_DG,Grid_Reward",
	})
	@property
	def weights(self):
		if not hasattr(self, "_weights"):
			self._weights = np.array([
				0, 0, 0, 0, 0, 0, 0, # reserved 0~6
				0., # knolwedge base retrieval score
				0.4, 0.6, # POStag 2 and 1
				# clue based score [0. bm25], deprecated:[1. TF-IDF cos, 2. Edit Distance Similarity, 3. Exact Match, 4. punctuation-free exact match]
				1, # "ClueMatch": 10,
				# word basic info:[0. whether is a complete word, 1. length of the word]
				5, # "IsAWord": 11,
				0, # "WordLen": 12,
				# word prior features: [0. #occurs from ClueDB, 1. #occurs from CrossWordGiant, 2. #occurs from unigram, 3. wiki title]
				0.12337192, # "OccCDB": 13, [ 1.2337192 ],
				0.021331637, # "OccCWG": 14, [ 0.21331637],
				-0.017480089, # "OccUNG": 15, [-0.17480089]
				0.03103069, # "InWiki": 16, [ 0.3103069 ]
				# clue-word score: [0. w2v similarity of word and clues, 1. bigram score for filling blanks (TODO), 2. pos tag (TODO)]
				0.8274433, # "W2Vcos": 17, [ 8.274433  ]
				0.0, # blank filling,
				0., # dictionary retrieval score
			])
			self._weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
				0.4, 0.6, 1.0, 15.0, 0.0, 
				0.8636034399999999, 0.149321459, -0.12236062299999999, 0.21721483, 5.792103099999999, \
				0.0, 0.0])
		return self._weights
	@property
	def bias(self): 
		if not hasattr(self, "_bias"):
			self._bias = 0
		return self._bias
	@property
	def initializations(self): return [
		self.build_tokenizers,
	]
	def load_data(self, dirn, tao):
		if type(dirn) == str: fns = [join(dirn, fn) for fn in os.listdir(dirn)]
		else: fns = dirn
		puzzles = []
		for fn in fns:
			with h5py.File(fn, "r") as dfile:
				X_feats = dfile["X_feats"][:]
				Y_scores = np.exp(dfile["Y_scores"][:]/tao)
			X_feats[:,12] = 0
			probs = Y_scores / Y_scores.sum()
			puzzles.append((X_feats, Y_scores, probs))
		return puzzles
	def data_gen(self, puzzles, batch_size, sample_size):
		n_feats = puzzles[0][0].shape[-1]
		X_batch = np.zeros((batch_size, sample_size, n_feats), dtype="float32")
		Y_batch = np.zeros((batch_size, sample_size, ), dtype="float32")
		while True:
			pzlids = np.random.choice(len(puzzles), batch_size).tolist()
			for i, k in enumerate(pzlids):
				X_feats, Y_scores, probs = puzzles[k]
				indices = np.random.choice(len(probs), sample_size, p=probs)
				X_batch[i] = X_feats[indices]
				Y_batch[i] = Y_scores[indices]
			yield X_batch, Y_batch
	def build_tokenizers(self, *pargs, **kwargs):
		self._tokenizers = []
		self._classes = []
	def build_model(self, *pargs, **kwargs):
		inp_feats = Input((self.n_features, ), dtype="float32")
		fc = Dense(1)
		oup_rwds = fc(inp_feats)
		oup_rwds = Lambda(lambda x:K.squeeze(x, -1))(oup_rwds)
		fc.set_weights([self.weights[:,None], np.array([self.bias])])
		self._pmodel = Model(inp_feats, oup_rwds)
		
		inp_feats_train = Input((self.sample_size, self.n_features), dtype="float32")
		oup_scores = self._pmodel(inp_feats_train)
		out_probs = Softmax()(oup_scores)
		
		self._model = Model(inp_feats_train, out_probs)
		loss_func = {
			"CE": crossentropy,
			"KL": kullback_leibler_divergence,
		}[self.loss]
		self._model.compile(Adam(1e-3), loss_func, metrics=["accuracy"])
		self._model.summary()
		# print(self._pmodel.get_weights()[0])
		# print(self._weights)
	def train(self, save_dir=None, dirn=None, \
		batch_size=64, tao=.1, val_ratio=0.1):
		if dirn is None: dirn = self.train_dir
		if save_dir is None: save_dir = self.default_model_dir
		if not hasattr(self, "_model"): self.build_model()

		fns = [join(dirn, fn) for fn in os.listdir(dirn)]
		random.shuffle(fns)
		vsize = int(val_ratio*len(fns))
		val_fns, tra_fns = fns[:vsize], fns[vsize:]
		print(len(tra_fns), "train files")
		tra_pzls = self.load_data(tra_fns, tao)
		tra_data_gen = self.data_gen(tra_pzls, batch_size, self.sample_size)
		print(len(val_fns), "valid files")
		val_pzls = self.load_data(val_fns, tao)
		val_data_gen = self.data_gen(val_pzls, batch_size*2, self.sample_size)
		self.valid_data = next(val_data_gen)
		print("valid X", self.valid_data[0].shape)

		before = self._model.get_weights()[0].flatten().tolist()
		logfile = os.path.join(save_dir, 'log.txt')
		try:
			self._model.fit(tra_data_gen, steps_per_epoch=500, epochs=20, validation_data = self.valid_data, max_queue_size=30, 
				callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.6, verbose=1, patience=2, min_lr=0, cooldown=2), 
					LogEvalSave(filename=logfile, caller=self, 
						save_dir=save_dir, select_rules=getattr(self, "select_rules", None))])
		except Exception: print("Interupted!")
		self.load()
		print("before", before)
		print("after", self._model.get_weights()[0].flatten().tolist())
	def predict(self, feats, **kwargs): return self._pmodel.predict(feats)
class ClfRewarder(Rewarder):
	'''
	Local Rewarder: training the reward function as a 0-1 classifier
	'''
	@property
	def params(self): return dict(super().params, **{
		"mask_ratio": 0.,
		"n_negs": 50,
		"lr": 0.001,
		"trainset": 0,
		"feat_norm": 0,
		"zero_init": 0,
		"fn": None,
		"fm": 0,
	})
	@property
	def options(self): return dict(super().options, **{
		"train_dirs": [
			"intermediate/saved_words/MCTS_DG,default.feats",#reward_feats/nyt.shuffle",
			"intermediate/saved_words/test_clues_retr.feats",
			"intermediate/saved_words/test_clues_retr_r1.feats",
			"intermediate/saved_words/test_clues_retr_r2.feats",
			"intermediate/saved_words/MCTS_DG_0713,default.feats",
		]
	})
	def feat_mask(self, source):
		if source > 0 or self.fm == 0:
			if not hasattr(self, "_ones"):
				self._ones = np.ones(20)
			return self._ones
		if not hasattr(self, "_feat_mask"):
			self._feat_mask = np.zeros(20)
			for i in [10, 18, 19, 7]: self._feat_mask[i] = 1
		return self._feat_mask
	def build_model(self, *pargs, **kwargs):
		inp_feats = Input((self.n_features, ), dtype="float32")
		fc = Dense(1)
		oup_rwds = fc(inp_feats)
		oup_rwds = Lambda(lambda x:K.squeeze(x, -1))(oup_rwds)
		fc.set_weights([self.weights[:,None], np.array([self.bias])])
		self._pmodel = Model(inp_feats, oup_rwds)
		oup_probs = Lambda(lambda x:K.sigmoid(x[:,None]))(oup_rwds)
		self._model = Model(inp_feats, oup_probs)
		self._model.compile(Adam(self.lr), 'binary_crossentropy', metrics=["accuracy"])
		self._model.summary()
	def load_data(self, dirn):
		if type(dirn) == str: fns = [join(dirn, fn) for fn in os.listdir(dirn)]
		else: fns = dirn
		X, Y = None, None
		for fn in fns:
			with h5py.File(fn, "r") as dfile:
				X_feats = dfile["features"][:]
				Y_labels = dfile["labels"][:]
			pos = []; neg = []
			for i, y in enumerate(Y_labels):
				if y[0] == 1: pos.append(i)
				else: neg.append(i)
			masked_indices = np.random.choice(pos, int(len(pos)*self.mask_ratio), False).tolist()
			if masked_indices:
				X_feats[masked_indices][10] = 0
			# if len(pos) * self.n_negs < len(neg):
			# 	neg = np.random.choice(neg, len(pos)*self.n_negs+1, False).tolist()
			Y_labels = Y_labels[pos+neg, 0]
			X_feats = X_feats[pos+neg]
			#X_feats[-1] = 0 # last negative feature: all 0
			if X is None:
				X, Y = X_feats, Y_labels
			else: 
				X = np.concatenate([X, X_feats], axis=0)
				Y = np.concatenate([Y, Y_labels], axis=0)
		print(X.shape, Y.shape)
		return [X], Y
	def h5data_path(self, dirn): return "%s.%s,%s.h5"%(dirn.rstrip("/"), type(self).__name__, self._banner({"loss", "lr"}))
	def train(self, save_dir=None, dirn=None, epochs=20,\
		batch_size=1024, tao=0.1, val_ratio=0.1):
		if dirn is None: dirn = self.train_dirs[self.trainset]
		if save_dir is None: save_dir = self.default_model_dir
		if not hasattr(self, "_model"): self.build_model()

		datafile = self.h5data_path(dirn)
		if datafile is not None and exists(datafile):
			print("loading from existsing h5 file", datafile)
			with h5py.File(datafile, "r") as dfile:
				trainX, validX = [], []
				for i in range(10):
					key = "trainX%d"%i
					if key in dfile:
						trainX.append(dfile[key][:])
					key = "validX%d"%i
					if key in dfile:
						validX.append(dfile[key][:])
				trainY = dfile["trainY"][:]
				validY = dfile["validY"][:]
			self.valid_data = (validX, validY)
		else:
			print("constructing new h5 file")
			fns = [join(dirn, fn) for fn in os.listdir(dirn)]
			vsize = int(val_ratio*len(fns))
			val_fns, tra_fns = fns[:vsize], fns[vsize:]
			print(len(tra_fns), "train files")
			trainX, trainY = self.load_data(tra_fns)
			print(len(val_fns), "valid files")
			self.valid_data = self.load_data(val_fns)
			for x in trainX:
				isnan = np.isnan(x)
				x[isnan] = 0
			for x in self.valid_data[0]:
				isnan = np.isnan(x)
				x[isnan] = 0
			if datafile is not None:
				with h5py.File(datafile, "w") as dfile:
					for i in range(len(trainX)):
						dfile.create_dataset("trainX%d"%i, data=trainX[i])
						dfile.create_dataset("validX%d"%i, data=self.valid_data[0][i])
					dfile.create_dataset("trainY", data=trainY)
					dfile.create_dataset("validY", data=self.valid_data[1])
		print("train X", [x.shape for x in trainX])
		print("valid X", [x.shape for x in self.valid_data[0]])
		self.fit(trainX, trainY, save_dir, batch_size, epochs)
		self.load()
	def fit(self, trainX, trainY, save_dir, batch_size, epochs):
		if self.feat_norm:
			self.fn = self.fnorm(trainX)
			for x in trainX: x /= self.fn
			for x in self.valid_data[0]: x /= self.fn
			print("feature normalization", self.fn)
			print("train X average")
			for x in trainX:
				print(x.mean(axis=0))
			print("valid X average")
			for x in self.valid_data[0]:
				print(x.mean(axis=0))
			weights, bias = self._pmodel.get_weights()
			if self.zero_init == 1:
				weights *= 0
			elif self.zero_init == 2:
				weights = np.zeros_like(weights)
				weights[10] = 1
			else:
				weights *= self.fn[:,None]
			print("init weightrs", weights.flatten())
			self._pmodel.set_weights([weights, bias])

		logfile = os.path.join(save_dir, 'log.txt')
		try:
			self._model.fit(trainX, trainY, batch_size=batch_size, epochs=epochs, max_queue_size=30, \
				validation_data = self.valid_data, callbacks=[ReduceLROnPlateau(monitor='val_loss', \
				factor=0.6, verbose=1, patience=2, min_lr=0, cooldown=2), LogEvalSave(filename=logfile, \
				caller=self, save_dir=save_dir, select_rules=getattr(self, "select_rules", None), save_init=True)])
		except KeyboardInterrupt: print("Interupted!")
		except Exception: traceback.print_exc()
	def fnorm(self, trainX): return trainX[0].mean(axis=0)+1e-9
	def predict(self, feats, source=1):
		mask = self.feat_mask(source)
		if self.fn is None: return self._pmodel.predict(feats*mask)
		else: return self._pmodel.predict(feats/self.fn*mask)
class RNRewarder(ClfRewarder): # Ranking model
	'''
	Local Rewarder: training the reward function as a ranking model
	'''
	@property
	def params(self): return dict(super().params, **{
		"loss": 'ranknet', # or hinge
		"margin": 10,
		"use_retr": 1, 
		"use_vocab": 1,
		"r_nw": 0., # non-word ratio in negative samples
	})
	def load_data(self, dirn):
		if type(dirn) == str: fns = [join(dirn, fn) for fn in os.listdir(dirn)]
		else: fns = dirn
		X_pos, X_neg = None, None
		for idx, fn in enumerate(fns):
			with h5py.File(fn, "r") as dfile:
				X_feats = dfile["features"][:]
				Y_labels = dfile["labels"][:]
			pos = defaultdict(list); neg = defaultdict(list)
			for ind, (y, i, num) in enumerate(Y_labels):
				if self.use_retr == 0 and num == 0: continue
				if self.use_vocab == 0 and num > 0: continue
				if y == 1: pos[i, num].append(ind)
				else: neg[i, num].append(ind)
			pos_inds, neg_inds = [], []
			for k in pos:
				ninds = neg[k]
				for p in pos[k]:
					if np.random.random() < self.mask_ratio: X_feats[p][10] = 0
					if self.n_negs < len(ninds):
						ninds = np.random.choice(ninds, self.n_negs+1, False).tolist()
					for n in ninds:
						pos_inds.append(p)
						neg_inds.append(n)
					#X_feats[ninds[-1]] = 0

			X_pos_batch, X_neg_batch = X_feats[pos_inds], X_feats[neg_inds]
			if X_pos is None:
				X_pos, X_neg = X_pos_batch, X_neg_batch
			else: 
				X_pos = np.concatenate([X_pos, X_pos_batch], axis=0)
				X_neg = np.concatenate([X_neg, X_neg_batch], axis=0)
			sys.stdout.write("\r%d/%d\t%s"%(idx+1, len(fns), fn))
			sys.stdout.flush()
			del X_neg_batch, X_pos_batch
		print(X_pos.shape)
		return [X_pos, X_neg], np.ones(len(X_pos))
	def build_model(self, *pargs, **kwargs):
		inp_feats = Input((self.n_features, ), dtype="float32")
		fc = Dense(1)
		oup_rwds = fc(inp_feats)
		oup_rwds = Lambda(lambda x:K.squeeze(x, -1))(oup_rwds)
		fc.set_weights([self.weights[:,None], np.array([self.bias])])
		self._pmodel = Model(inp_feats, oup_rwds)
		inp_feats_neg = Input((self.n_features, ), dtype="float32")
		oup_rwds_neg = self._pmodel(inp_feats_neg)
		if self.loss == "ranknet":
			oups = Lambda(lambda x:K.sigmoid(x[0]-x[1])[:,None])([oup_rwds, oup_rwds_neg])
			self._model = Model([inp_feats, inp_feats_neg], oups)
			self._model.compile(Adam(self.lr), 'binary_crossentropy', metrics=["accuracy"])
		elif self.loss == "hinge":
			oups = Lambda(lambda x:x[0][:,None]-x[1][:,None])([oup_rwds, oup_rwds_neg])
			self._model = Model([inp_feats, inp_feats_neg], oups)
			loss_func, acc_func = contrastive(float(self.margin))
			self._model.compile(Adam(self.lr), loss_func, metrics=[acc_func])
		self._model.summary()
class SMRewarder(RNRewarder): # Softmax loss
	@property
	def params(self): return dict(super().params, **{
		"margin": 10,
		"use_retr": 1, 
		"use_vocab": 1,
		"wl": 0,
		"epsilon": 1e-7,
		"smooth": 0.,
		"temp": 1.,
	})
	def fnorm(self, trainX): 
		return trainX[0][:,0,:].mean(axis=0)+1e-9
	def h5data_path(self, dirn): return None		
	def load_data(self, dirn):
		if type(dirn) == str: fns = [join(dirn, fn) for fn in sorted(os.listdir(dirn))]
		else: fns = dirn
		X = None
		np.random.seed(12318)
		nnn = int(self.n_negs*self.r_nw)
		n_negs = self.n_negs - nnn
		for idx, fn in enumerate(fns):
			with h5py.File(fn, "r") as dfile:
				X_feats = dfile["features"][:]
				Y_labels = dfile["labels"][:]
			pos = defaultdict(list); neg = defaultdict(list)
			for ind, (y, i, num) in enumerate(Y_labels):
				if self.use_retr == 0 and num == 0: continue
				if self.use_vocab == 0 and num > 0: continue
				if y == 1: 
					pos[i, num].append(ind)
					
				else: neg[i, num].append(ind)

			X_batch = []
			for k in pos:
				ninds = neg[k]
				if ninds:
					for p in pos[k]:
						#if np.random.random() < self.mask_ratio: X_feats[p][10] = 0
						if n_negs < len(ninds):
							ninds = np.random.choice(ninds, n_negs, False).tolist()
						elif n_negs > len(ninds):
							ninds = np.random.choice(ninds, n_negs, True).tolist()
						ind_batch = [p] + ninds
						x_batch = X_feats[ind_batch]*self.feat_mask(k[1])
						if nnn:
							x_batch = np.concatenate([x_batch, np.zeros((nnn, x_batch.shape[1]))], axis=0)
						X_batch.append(x_batch)
						# if k[1]: # from vocab
						# 	X_batch.append(X_feats[ind_batch])
						# else:
						# 	X_batch.append(X_feats[ind_batch]*feat_mask)
					
			X_batch = np.array(X_batch)
			if X is None:
				X = X_batch
			else: 
				X = np.concatenate([X, X_batch], axis=0)
			sys.stdout.write("\r%d/%d\t%s"%(idx+1, len(fns), fn))
			sys.stdout.flush()
			del X_batch
		print(X.shape)
		return [X], [np.zeros(len(X))[:,None]]	
	def build_model(self, *pargs, **kwargs):
		inp_feats = Input((self.n_negs+1, self.n_features, ), dtype="float32")
		fc = Dense(1)
		oup_rwds = fc(inp_feats)
		oup_rwds = Lambda(lambda x:K.squeeze(x, -1))(oup_rwds)
		fc.set_weights([self.weights[:,None], np.array([self.bias])])
		
		if self.wl:
			self._model = Model(inp_feats, oup_rwds)
			self._model.compile(Adam(self.lr), weighted_CE(self.epsilon, self.temp, self.smooth), metrics=["accuracy"])
		else:
			oup_probs = Softmax()(oup_rwds)
			self._model = Model(inp_feats, oup_probs)
			self._model.compile(Adam(self.lr), 'sparse_categorical_crossentropy', metrics=[MRR, "accuracy"])
		self._model.summary()

		inps = Input((self.n_features, ), dtype="float32")
		oups = fc(inps)
		oups = Lambda(lambda x:K.squeeze(x, -1))(oups)
		self._pmodel = Model(inps, oups)
	def fit(self, trainX, trainY, save_dir, batch_size, epochs):
		if self.feat_norm:
			self.fn = self.fnorm(trainX)
			for x in trainX: x /= self.fn
			for x in self.valid_data[0]: x /= self.fn
			print("feature normalization", self.fn)
			print("train X average")
			for x in trainX:
				print(x[:,0,:].mean(axis=0))
				print(x[:,1:,:].mean(axis=0).mean(axis=0))
			print("valid X average")
			for x in self.valid_data[0]:
				print(x[:,0,:].mean(axis=0))
				print(x[:,1:,:].mean(axis=0).mean(axis=0))
			weights, bias = self._pmodel.get_weights()
			if self.zero_init == 1:
				weights *= 0
			elif self.zero_init == 2:
				weights = np.zeros_like(weights)
				weights[10] = 1
			else:
				weights *= self.fn[:,None]
			print("init weightrs", weights.flatten())
			self._pmodel.set_weights([weights, bias])

		logfile = os.path.join(save_dir, 'log.txt')
		try:
			self._model.fit(trainX, trainY, batch_size=batch_size, epochs=epochs, max_queue_size=30, \
				validation_data = self.valid_data, callbacks=[ReduceLROnPlateau(monitor='val_loss', \
				factor=0.6, verbose=1, patience=2, min_lr=0, cooldown=2), LogEvalSave(filename=logfile, \
				caller=self, save_dir=save_dir, select_rules=getattr(self, "select_rules", None), save_init=True)])
		except KeyboardInterrupt: print("Interupted!")
		except Exception: traceback.print_exc()
class SMR_MLP(SMRewarder):
	@property
	def params(self): return dict(super().params, **{
		"fn": None,
		"lr": 0.01,
		"hiddens": "128",
		"l2": 0,
	})
	def fit(self, trainX, trainY, save_dir, batch_size, epochs):
		self.fn = self.fnorm(trainX)
		for x in trainX: x /= self.fn
		for x in self.valid_data[0]: x /= self.fn
		print("feature normalization", self.fn)
		print("train X average")
		for x in trainX:
			print(x[:,0,:].mean(axis=0))
		print("valid X average")
		for x in self.valid_data[0]:
			print(x[:,0,:].mean(axis=0))
		logfile = os.path.join(save_dir, 'log.txt')
		try:
			self._model.fit(trainX, trainY, batch_size=batch_size, epochs=epochs, max_queue_size=30, \
				validation_data = self.valid_data, callbacks=[ReduceLROnPlateau(monitor='val_loss', \
				factor=0.6, verbose=1, patience=2, min_lr=0, cooldown=2), LogEvalSave(filename=logfile, \
				caller=self, save_dir=save_dir, select_rules=getattr(self, "select_rules", None), save_init=True)])
		except KeyboardInterrupt: print("Interupted!")
		except Exception: traceback.print_exc()
	def build_model(self, *pargs, **kwargs):
		x = inp_feats = Input((self.n_negs+1, self.n_features, ), dtype="float32")
		fcs = [Dense(int(l), activation="tanh", kernel_regularizer=l2() if self.l2 else None) for l in self.hiddens.split("_")] + [Dense(1)]
		for fc in fcs:
			x = fc(x)
			x = Dropout(0.2)(x)
		oup_rwds = Lambda(lambda x:K.squeeze(x, -1))(x)
		#fc.set_weights([self.weights[:,None], np.array([self.bias])])
		
		if self.wl:
			self._model = Model(inp_feats, oup_rwds)
			self._model.compile(Adam(self.lr), weighted_CE(self.epsilon, self.temp), metrics=[MRR, "accuracy"])
		else:
			oup_probs = Softmax()(oup_rwds)
			self._model = Model(inp_feats, oup_probs)
			self._model.compile(Adam(self.lr), 'sparse_categorical_crossentropy', metrics=[MRR, "accuracy"])
		
		x = inps = Input((self.n_features, ), dtype="float32")
		for fc in fcs:
			x = fc(x)
		oups = Lambda(lambda x:K.squeeze(x, -1))(x)
		self._pmodel = Model(inps, oups)
class CTRewarder(ClfRewarder): # Constrastive Rewarder
	'''
	Local Rewarder: training the reward function with constrastive objective
		margin loss for pos and neg sample
		l1 sim loss for pos and pos sample
	'''
	@property
	def params(self): return dict(super().params, **{
		"margin": 10,
		"n_pos": 1,
		"n_nw": 1,
	})
	def load_data(self, dirn):
		if type(dirn) == str: fns = [join(dirn, fn) for fn in os.listdir(dirn)]
		else: fns = dirn
		X_pos, X_neg, Y = [], [], []
		for idx, fn in enumerate(fns):
			with h5py.File(fn, "r") as dfile:
				X_feats = dfile["features"][:]
				Y_labels = dfile["labels"][:]
			pos = defaultdict(list); neg = defaultdict(list)
			for ind, (y, i, num) in enumerate(Y_labels):
				if y == 1: pos[i, num].append(ind)
				else: neg[i, num].append(ind)
			pos_inds, neg_inds, labels = [], [], []
			for k in pos:
				ninds = neg[k]
				for p in pos[k]:
					if np.random.random() < self.mask_ratio: X_feats[p][10] = 0
					if self.n_negs < len(ninds):
						ninds = np.random.choice(ninds, self.n_negs, False).tolist()
					for n in ninds:
						pos_inds.append(p)
						neg_inds.append(n)
						labels.append(1)
					# for i in range(self.n_nw):
					# 	X_feats[ninds[-i]] = 0
			allpos = []
			for v in pos.values(): allpos += v
			for _ in range(self.n_pos*len(allpos)):
				i1, i2 = np.random.choice(allpos, 2, False).tolist()
				pos_inds.append(i1)
				neg_inds.append(i2)
				labels.append(0)

			X_pos.append(X_feats[pos_inds])
			X_neg.append(X_feats[neg_inds])
			Y.append(np.array(labels))

			sys.stdout.write("\r%d/%d\t%s"%(idx+1, len(fns), fn))
			sys.stdout.flush()
		X_pos = np.concatenate(X_pos, axis=0)
		X_neg = np.concatenate(X_neg, axis=0)
		Y = np.concatenate(Y, axis=0)
		print(X_pos.shape)
		return [X_pos, X_neg], Y
	def build_model(self, *pargs, **kwargs):
		inp_feats = Input((self.n_features, ), dtype="float32")
		fc = Dense(1)
		oup_rwds = fc(inp_feats)
		oup_rwds = Lambda(lambda x:K.squeeze(x, -1))(oup_rwds)
		fc.set_weights([self.weights[:,None], np.array([self.bias])])
		self._pmodel = Model(inp_feats, oup_rwds)
		inp_feats_neg = Input((self.n_features, ), dtype="float32")
		oup_rwds_neg = self._pmodel(inp_feats_neg)
		oups = Lambda(lambda x:x[0][:,None]-x[1][:,None])([oup_rwds, oup_rwds_neg])
		self._model = Model([inp_feats, inp_feats_neg], oups)
		loss_func, acc_func = contrastive(float(self.margin))
		self._model.compile(Adam(self.lr), loss_func, metrics=[acc_func])
		self._model.summary()
class MicroCTR(CTRewarder):
	'''
	Local Rewarder: same model and loss but separate groups by SOURCE
	'''
	@property
	def params(self): return dict(super().params, **{
		"n_negs": 100,
		"mask_ratio": 0.,
	})
	@property
	def options(self): return dict(super().options, **{
		"train_dir": "intermediate/saved_words/MCTS_DG1,default.feats",
	})
	@property
	def h5data_path(self): return join("intermediate/saved_words", "%s,%s.h5"%(type(self).__name__, self._banner({"loss"})))
	def load_data(self, dirn):
		if type(dirn) == str: fns = [join(dirn, fn) for fn in os.listdir(dirn)]
		else: fns = dirn
		X_pos, X_neg, Y = [], [], []
		for idx, fn in enumerate(fns):
			with h5py.File(fn, "r") as dfile:
				X_feats = dfile["features"][:]
				Y_labels = dfile["labels"][:]
			if len(X_feats) == 0: continue
			pos = defaultdict(list); neg = defaultdict(list)
			for ind, (y, clueid, ptnid) in enumerate(Y_labels):
				if y == 1: pos[clueid, ptnid].append(ind)
				else: neg[clueid, ptnid].append(ind)
			pos_inds, neg_inds, labels = [], [], []
			for k in pos:
				ninds = neg[k]
				for p in pos[k]:
					if self.n_negs < len(ninds):
						ninds = np.random.choice(ninds, self.n_negs, False).tolist()
					for n in ninds:
						pos_inds.append(p)
						neg_inds.append(n)
						labels.append(1)
			allpos = []
			for v in pos.values(): allpos += v
			if len(allpos) > 1:
				for _ in range(self.n_pos*len(allpos)):
					i1, i2 = np.random.choice(allpos, 2, False).tolist()
					pos_inds.append(i1)
					neg_inds.append(i2)
					labels.append(0)

			X_pos.append(X_feats[pos_inds])
			X_neg.append(X_feats[neg_inds])
			Y.append(np.array(labels))

			sys.stdout.write("\r%d/%d\t%s"%(idx+1, len(fns), fn))
			sys.stdout.flush()
		X_pos = np.concatenate(X_pos, axis=0)
		X_neg = np.concatenate(X_neg, axis=0)
		Y = np.concatenate(Y, axis=0)
		print(X_pos.shape)
		return [X_pos, X_neg], Y

method_list = dict(
	m for m in inspect.getmembers(sys.modules[__name__]) 
	if inspect.isclass(m[1]) and issubclass(m[1], Rewarder)
)