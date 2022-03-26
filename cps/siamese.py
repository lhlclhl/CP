import numpy as np, os, time, json, argparse, math
from os.path import join, exists
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
import keras.backend as K
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.metrics import AUC
from keras.losses import hinge
from keras_preprocessing.sequence import pad_sequences
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import to_array
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr

from .base import BaseModel, DataGenerator
from .puz_utils import string2fill

class Siamese(BaseModel):
	@property
	def params(self): return dict(super().params, **{
		"bert": "wwm_uncased",
		"n_epochs": 20,
		"split_training_data": 10, 
		"batch_size": 256,
		"n_fine_tune_vars": 8,
		"n_neg_samples": 10,
		"hidden_dims": 512,
		"learning_rate": 1e-3,
		"valid_ratio": 0.05,
		"heldout_ratio": 0.05,
	})
	@property
	def options(self): return dict(super().options, **{
		"buf_dir": "intermediate/clue_ver_data",
		"model_dir": "models/ranking_answers",
		"gen_dir": "./intermediate",
		"h5buf": 1,
		"dev_ratio": 0.2,
		"max_len": 32,
		"bert_path": "../bert/uncased_L-8_H-512_A-8",
	})
	@property
	def lr_schedule(self):
		return {
			0: 1,
			10000: 1,
			20000: 0.1
		}
	def load_data(self, datafile, limit=0):
		samples, labels = [], []
		with open(datafile, encoding="utf-8") as fin:
			for line in fin:
				item = json.loads(line.strip())
				for i in range(2):
					for k, v in item["candidates"][i].items():
						samples.append(v)
						labels.append((item["clues"][i][k][0], item["answers"][i][k]))
				limit -= 1
				if limit == 0: break
		print(len(samples), "Samples Loaded")
		return samples, labels
	def gen_h5_filename(self, filn):
		params = "%s,max_len=%d"%(type(self).__name__, self.max_len)
		return join(self.buf_dir, "%s,%s.h5"%(os.path.split(filn)[1], params)) if self.h5buf else None
	def build_tokenizers(self, data=None):
		self._tokenizers = []
		self._classes = []
		self._data_generator = lambda x,y:(x,y)
		self._tknz = Tokenizer(join(self.bert_path, "vocab.txt"), do_lower_case="uncased" in self.bert)
	def make_feature(self, samples, verbose=False):
		X_tokens, X_segments = [], []
		for s1, s2 in samples:
			tokens, segments = self._tknz.encode(s1, s2, maxlen=self.max_len)
			X_tokens.append(tokens)
			X_segments.append(segments)
		return [pad_sequences(X_tokens, padding="post"), pad_sequences(X_segments, padding="post")]
	def make_training_data(self, data):
		if not hasattr(self, "_tokenizers"): self.build_tokenizers(data)
		X, Y = [], []
		for cands, (clue, answer) in zip(*data):
			for cand, strings, score in cands:
				Y.append(int(answer in strings))
				X.append((clue, cand))
		X = self.make_feature(X)
		Y = np.array(Y)
		print("data shape", [x.shape for x in X])
		print("positive samples", (Y==1).sum())
		print("negative samples", (Y==0).sum())
		return X, [Y]
	def build_model(self):
		bert = build_transformer_model(
			config_path=join(self.bert_path, "bert_config.json"), 
			checkpoint_path=join(self.bert_path, "bert_model.ckpt"), 
			return_keras_model=False,
		)  

		
		output = Lambda(lambda x: x[:, 0], name='Output-CLS-token')(bert.model.output)
		if self.hidden_dims:
			output = Dense(
				units=self.hidden_dims,
				activation='relu',
				kernel_initializer=bert.initializer,
				name = "Output-Hidden"
			)(output)
			output = Dropout(0.2)(output)
		score = output = Dense(
			units=1,
			activation=None,
			kernel_initializer=bert.initializer,
			name = "Output-Classifier"
		)(output)
		pmodel = Model(bert.model.input, score)
		output = Activation("sigmoid")(output)

		model = Model(bert.model.input, output)
	
		for l in model.layers:
			if not l.name.startswith("Output"):
				l.trainable = False
		if self.n_fine_tune_vars:
			layers = [layer for layer in bert.model.layers if layer.name.startswith('Transformer-')]
			for i in range(self.n_fine_tune_vars):
				print("Unfrozen layer: ", layers[-1-i].name)
				layers[-1-i].trainable = True
		model.summary()

		AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')
		optim = AdamLR(learning_rate=self.learning_rate, lr_schedule=self.lr_schedule)
		model.compile(
			loss='binary_crossentropy',
			optimizer=optim,
			metrics=['accuracy', AUC(name='auc')],
		)
		self._pmodel = model
		self._model = model
def marginal_accuracy(margin=0.5):
	def accuracy(y_true, y_pred):
		if not K.is_tensor(y_pred):
			y_pred = K.constant(y_pred)
		y_true = K.cast(y_true, y_pred.dtype)
		return K.cast(
			K.maximum(margin-y_pred, 0) / margin, 
		K.floatx()) # when s_pos==s_neg, acc=0; when s_pos-s_neg==margin, acc=1
	return accuracy
class Marginal(Siamese):
	@property
	def params(self): return dict(super().params, **{
		"batch_size": 64,
		"fix_layers": 6,
		"margin": 0.05,
	})
	@property
	def lr_schedule(self):
		return {
			0: 1,
			100000: 1,
			200000: 0.1
		}
	def make_data(self, data, verbose=True, h5file=None, limit=0, shuffle=False):
		if not hasattr(self, "_tokenizers"): self.build_tokenizers(data)
		X, Y = [], []
		n_pos = n_neg = n_samples = 0
		for cands, (clue, answer) in zip(*data):
			X_batch, Y_batch = [], []
			fclue = string2fill(clue)
			for cand, strings, score in cands:
				if string2fill(cand) != fclue:
					Y_batch.append(int(answer in strings))
					X_batch.append((clue, cand))
			Y_batch = np.array(Y_batch)
			if len(Y_batch) > Y_batch.sum() > 0: # batch with positive and negative samples
				X.append(self.make_feature(X_batch))
				Y.append(Y_batch)
				n_samples += len(Y_batch)
				n_pos += (Y_batch==1).sum()
				n_neg += (Y_batch==0).sum()
		print("num of batches", len(X))
		print("average batch size", n_samples/len(Y))
		print("positive samples", n_pos)
		print("negative samples", n_neg)
		return BatchGenerator(X, Y)	
	def build_model(self, verbose=True):
		bert = build_transformer_model(
			config_path=join(self.bert_path, "bert_config.json"), 
			checkpoint_path=join(self.bert_path, "bert_model.ckpt"), 
			return_keras_model=False,
		)  

		output = Lambda(lambda x: x[:, 0], name='Output-CLS-token')(bert.model.output)
		if self.hidden_dims:
			output = Dense(
				units=self.hidden_dims,
				activation="relu",
				kernel_initializer=bert.initializer,
				name = "Output-Hidden"
			)(output)
			output = Dropout(0.2)(output)
		score = Dense(
			units=1,
			activation=None,
			kernel_initializer=bert.initializer,
			name = "Output-Classifier"
		)(output)
		pmodel = Model(bert.model.input, score)

		inp_labels = Input((1,), dtype="float32")
		neg_score = Lambda(lambda x: K.max(x[0]-x[1]*1e9, axis=0, keepdims=True))([score, inp_labels])
		output = Lambda(lambda x: K.maximum((self.margin-x[0]+x[1])*x[2], 0))([score, neg_score, inp_labels])
		model = Model(bert.model.input+[inp_labels], output)
	
		for layer in model.layers:
			if "Transformer" in layer.name and int(layer.name.split("-")[1]) >= self.fix_layers: 
				layer.trainable = True
			elif layer.name.startswith("Output"): layer.trainable = True
			else: layer.trainable = False

		model.compile(extend_with_piecewise_linear_lr(Adam, name='AdamLR')(
			learning_rate=self.learning_rate, lr_schedule=self.lr_schedule
		), loss=lambda tar,out:out, metrics=[marginal_accuracy(self.margin)])
		if verbose: model.summary()

		self._pmodel = pmodel
		self._model = model
class BatchGenerator(DataGenerator):
	def __init__(self, X, Y, **kwargs):
		for k, v in kwargs.items(): setattr(self, k, v)
		self.X = X if type(X) in {list, tuple} else [X]
		self.Y = Y if Y is None or type(Y) in {list, tuple} else [Y]
		self.steps = len(self.X)

	def __iter__(self, shuffle=False):
		inds = np.arange(self.steps).tolist()
		if shuffle: np.random.shuffle(inds)
		for i in inds:
			X_tokens_batch, X_segments_batch = self.X[i]
			labels_batch = self.Y[i]
			yield [X_tokens_batch, X_segments_batch, labels_batch], labels_batch
class Contrastive(Siamese):
	@property
	def params(self): return dict(super().params, **{
		"batch_size": 64,
		"fix_layers": 4,
		"sample_weight": 0,
		"fix_embs": 1,
	})
	def build_model(self, verbose=True):
		bert = build_transformer_model(
			config_path=join(self.bert_path, "bert_config.json"), 
			checkpoint_path=join(self.bert_path, "bert_model.ckpt"), 
			return_keras_model=False,
		)
		cls_emb = Lambda(lambda x:x[:, 0])(bert.model.output)
		self._encoder = Model(bert.model.input, cls_emb)

		for layer in self._encoder.layers:
			if "Transformer" in layer.name and int(layer.name.split("-")[1]) >= self.fix_layers: 
				layer.trainable = True
			elif "Embedding" in layer.name and self.fix_embs == 0:
				layer.trainable = True
			else: layer.trainable = False
		inputs_key = []
		inputs_value = []
		for inp in self._encoder.input:
			inputs_key.append(Input((None, ), dtype="float32", name=inp.name.split(":")[0]+"_key"))
			inputs_value.append(Input((None, ), dtype="float32", name=inp.name.split(":")[0]+"_value"))
		inputs = inputs_key + inputs_value
		key_encod = self._encoder(inputs_key)
		value_encod = self._encoder(inputs_value)

		encod_score = Lambda(lambda x:K.sum(x[0]*x[1], axis=-1))([key_encod, value_encod])
		self._pmodel = Model(inputs, encod_score) 

		encod_dot = Lambda(lambda x:K.dot(x[0], K.transpose(x[1])))([key_encod, value_encod])
		out_prob = Softmax(axis=-1)(encod_dot)
		self._model = Model(inputs, out_prob)
		self._model.compile(extend_with_piecewise_linear_lr(Adam, name='AdamLR')(
			learning_rate=self.learning_rate, lr_schedule=self.lr_schedule
		), "categorical_crossentropy", ["accuracy"])
		if verbose: self._model.summary()
	def load_data(self, datafile, limit=0, thres=0.1):
		clues = []
		clue2id = {}
		glosses = []
		gloss2id = {}
		hits = []
		with open(datafile, encoding="utf-8") as fin:
			for line in fin:
				score, clue, gloss = line.strip("\r\n").split("\t")
				if float(score) >= thres:
					if clue not in clue2id:
						clue2id[clue] = len(clues)
						clues.append(clue)
					if gloss not in gloss2id:
						gloss2id[gloss] = len(glosses)
						glosses.append(gloss)
					sw = min(1, 1 - self.sample_weight + self.sample_weight * float(score))
					hits.append((sw, [clue2id[clue]], [gloss2id[gloss]]))
		del clue2id, gloss2id
		return (clues, glosses), hits
	def split_dev(self, data, seed=12138):
		np.random.seed(12138)
		(clues, glosses), hits = data
		tl, dl = [], []
		for clues_and_glosses in hits:
			if np.random.random() < self.dev_ratio:
				dl.append(clues_and_glosses)
			else: 
				tl.append(clues_and_glosses)
		return ((clues, glosses), tl), ((clues, glosses), dl)
	def make_feature(self, samples, verbose=False):
		K_tokens, K_segments, V_tokens, V_segments = [], [], [], []
		for s1, s2 in samples:
			tokens, segments = self._tknz.encode(s1, maxlen=self.max_len)
			K_tokens.append(tokens)
			K_segments.append(segments)
			tokens, segments = self._tknz.encode(s2, maxlen=self.max_len)
			V_tokens.append(tokens)
			V_segments.append(segments)
		return [pad_sequences(K_tokens, padding="post", maxlen=self.max_len), pad_sequences(K_segments, padding="post", maxlen=self.max_len), \
			pad_sequences(V_tokens, padding="post", maxlen=self.max_len), pad_sequences(V_segments, padding="post", maxlen=self.max_len)]
	def make_data(self, data, verbose=True, h5file=None, limit=0, shuffle=False):
		(clues, glosses), hits = data
		vclues, vglosses = set(), set()
		for _, hclues, hglosses in hits:
			for c in hclues: vclues.add(c)
			for g in hglosses: vglosses.add(g)
		for i in vclues:
			if type(clues[i]) == str: clues[i], _ = self._tknz.encode(clues[i], maxlen=self.max_len)
		for i in vglosses:
			if type(glosses[i]) == str: glosses[i], _ = self._tknz.encode(glosses[i], maxlen=self.max_len)
		return PairGenerator(clues, glosses, hits, batch_size=self.batch_size)
def mrr(y_true, y_pred):
	y_tar = K.sum(y_true * y_pred, axis=-1, keepdims=True)
	return 1/(K.sum(K.cast(K.greater(y_pred, y_tar), "float32"), axis=-1)+1)
class ContrastiveMixup(Contrastive):
	@property
	def params(self): return dict(super().params, **{
		"batch_size": 32, # batch size of positive samples
		"n_negs": 4,
		"fix_layers": 6,
	})
	@property
	def options(self): return dict(super().options, **{
		"select_rules": [("mrr", -1)],
	})
	def make_feature(self, samples, verbose=False):
		K_offsets, V_tokens, V_segments = [], [], []
		last_key = None
		for s1, s2 in samples:
			if s1 != last_key:
				last_key = s1
				K_offsets.append(0)
				tokens, segments = self._tknz.encode(s1, maxlen=self.max_len)
				V_tokens.append([tokens])
				V_segments.append([segments])
			tokens, segments = self._tknz.encode(s2, maxlen=self.max_len)
			V_tokens[-1].append(tokens)
			V_segments[-1].append(segments)
		ret = []
		for i in range(len(K_offsets)):
			ret.append( [pad_sequences(V_tokens[i], padding="post", maxlen=self.max_len)[None,:,:], 
				pad_sequences(V_segments[i], padding="post", maxlen=self.max_len)[None,:,:], 
				np.array(K_offsets[i:i+1])] )
		return ret
	def do_batch(self, samples, norm=1):
		featss = self.make_feature(samples)
		ret = []
		for feats in featss:
			scores = self.predict(feats)[0][0]
			if norm: 
				# scores = np.exp(scores - scores[0]) # norm with exp
				scores /= scores[0]
			ret += scores.tolist()[1:]
		# K_offsets = feats[-1]
		# for i, offset in enumerate(K_offsets):``
		# 	end = len(scores) if i+1 == len(K_offsets) else K_offsets[i+1]
		# 	for k in range(offset+1, end): ret.append(scores[k])
		assert len(ret) == len(samples), "n_samples (%d) != n_rets (%d)" % (len(samples), len(ret))
		return [np.array(ret)]
	def build_model(self, verbose=True):
		bert = build_transformer_model(
			config_path=join(self.bert_path, "bert_config.json"), 
			checkpoint_path=join(self.bert_path, "bert_model.ckpt"), 
			return_keras_model=False,
		)
		cls_emb = Lambda(lambda x:x[:, 0])(bert.model.output)
		self._encoder = Model(bert.model.input, cls_emb)

		for layer in self._encoder.layers:
			if "Transformer" in layer.name and int(layer.name.split("-")[1]) >= self.fix_layers: 
				layer.trainable = True
			elif "Embedding" in layer.name and self.fix_embs == 0:
				layer.trainable = True
			else: layer.trainable = False
		
		inputs = []
		for inp in self._encoder.input:
			inputs.append(Input((None, None), dtype="float32", name=inp.name.split(":")[0]+"_multi"))
		input_keys = Input((1,), dtype="int32")

		inputs_flatten = [Lambda(lambda x: 
			tf.reshape(x, tf.stack([K.prod(K.shape(x)[:-1]), K.shape(x)[-1]])),
		)(inpv) for inpv in inputs]
		encods = self._encoder(inputs_flatten)

		key_encod = Lambda(lambda x: K.gather(x[0], x[1][:,0]),
			#K.dot(K.one_hot(x[1], K.shape(x[0])[0]), x[0]),
			name="gather_keys")([encods, input_keys])
		value_encod = encods

		# encod_score = Lambda(lambda x:K.sum(x[0]*x[1], axis=-1))([key_encod, value_encod])
		# self._pmodel = Model(inputs, encod_score) 

		encod_dot = Lambda(lambda x:K.dot(x[0], K.transpose(x[1])), name="calc_sim")([key_encod, value_encod])
		self._pmodel = Model(inputs+[input_keys], encod_dot) 

		out_prob = Lambda(lambda x: K.softmax(
			x[0] - (x[0]+1e9) * K.one_hot(x[1][:,0], K.shape(x[0])[1]), 
		axis=-1), name="out_prob")([encod_dot, input_keys]) # eliminate themselves
		self._model = Model(inputs+[input_keys], out_prob)
		self._model.compile(extend_with_piecewise_linear_lr(Adam, name='AdamLR')(
			learning_rate=self.learning_rate, lr_schedule=self.lr_schedule
		), "categorical_crossentropy", ["accuracy", mrr])
		if verbose: self._model.summary()
	def load_data(self, datafile, limit=0, thres=0.1):
		clues = []
		clue2id = {}
		samples = []
		with open(datafile, encoding="utf-8") as fin:
			for line in fin:
				item = json.loads(line)
				clue = item["clue"]
				pclues = item["positive"]
				nclues = item["negative"]
				for c in [clue] + pclues + nclues:
					if c not in clue2id:
						clue2id[c] = len(clues)
						clues.append(c)
				samples.append((clue2id[clue], [clue2id[c] for c in pclues], [clue2id[c] for c in nclues]))
		del clue2id
		return clues, samples	
	def split_dev(self, data, seed=12138):
		np.random.seed(12138)
		clues, samples = data
		tl, dl = [], []
		for s in samples:
			if np.random.random() < self.dev_ratio:
				dl.append(s)
			else: 
				tl.append(s)
		return (clues, tl), (clues, dl)
	def make_data(self, data, verbose=True, h5file=None, limit=0, shuffle=False):
		clues, samples = data
		vclues = set()
		for clue, pos, negs in samples:
			vclues.add(clue)
			vclues |= set(pos)
			vclues |= set(negs)
		for i in vclues:
			if type(clues[i]) == str: clues[i], _ = self._tknz.encode(clues[i], maxlen=self.max_len)
		return PNGenerator(clues, samples, batch_size=self.batch_size, n_negs=self.n_negs)
class ContrastiveMixupM(ContrastiveMixup):
	@property
	def params(self): return dict(super().params, **{
		"batch_size": 256, # batch size of positive samples
		"n_negs": 2,
	})
	def make_data(self, data, verbose=True, h5file=None, limit=0, shuffle=False):
		clues, samples = data
		vclues = set()
		for clue, pos, negs in samples:
			vclues.add(clue)
			vclues |= set(pos)
			vclues |= set(negs)
		for i in vclues:
			if type(clues[i]) == str: clues[i], _ = self._tknz.encode(clues[i], maxlen=self.max_len)
		return MUGenerator(clues, samples, batch_size=self.batch_size, n_negs=self.n_negs)
class CTRMMM(ContrastiveMixupM):
	@property
	def params(self): return dict(super().params, **{
		"batch_size": 340, # batch size of positive samples
		"n_negs": 1,
	})
class ContrastiveMarginal(Contrastive):
	@property
	def params(self): return dict(super().params, **{
		"fix_layers": 6,
		"margin": 0.1,
	})
	def make_data(self, data, verbose=True, h5file=None, limit=0, shuffle=False):
		(clues, glosses), hits = data
		vclues, vglosses = set(), set()
		for _, hclues, hglosses in hits:
			for c in hclues: vclues.add(c)
			for g in hglosses: vglosses.add(g)
		for i in vclues:
			if type(clues[i]) == str: clues[i], _ = self._tknz.encode(clues[i], maxlen=self.max_len)
		for i in vglosses:
			if type(glosses[i]) == str: glosses[i], _ = self._tknz.encode(glosses[i], maxlen=self.max_len)
		return PairGenerator(clues, glosses, hits, batch_size=self.batch_size, y_input=True)
	def build_model(self, verbose=True):
		bert = build_transformer_model(
			config_path=join(self.bert_path, "bert_config.json"), 
			checkpoint_path=join(self.bert_path, "bert_model.ckpt"), 
			return_keras_model=False,
		)
		cls_emb = Lambda(lambda x:x[:, 0])(bert.model.output)
		self._encoder = Model(bert.model.input, cls_emb)

		for layer in self._encoder.layers:
			if "Transformer" in layer.name and int(layer.name.split("-")[1]) >= self.fix_layers: 
				layer.trainable = True
			elif "Embedding" in layer.name and self.fix_embs == 0:
				layer.trainable = True
			else: layer.trainable = False
		inputs_key = []
		inputs_value = []
		for inp in self._encoder.input:
			inputs_key.append(Input((None, ), dtype="float32", name=inp.name.split(":")[0]+"_key"))
			inputs_value.append(Input((None, ), dtype="float32", name=inp.name.split(":")[0]+"_value"))
		inputs = inputs_key + inputs_value
		key_encod = self._encoder(inputs_key)
		value_encod = self._encoder(inputs_value)

		encod_score = Lambda(lambda x:K.sum(x[0]*x[1], axis=-1))([key_encod, value_encod])
		self._pmodel = Model(inputs, encod_score) 

		encod_dot = Lambda(lambda x:K.dot(x[0], K.transpose(x[1])))([key_encod, value_encod])
		inp_labels = Input((None, ), dtype="float32")
		out_pos = Lambda(lambda x: K.mean(x[0]*x[1], axis=-1))([encod_dot, inp_labels])
		out_neg = Lambda(lambda x: x[0]*(1-x[1]))([encod_dot, inp_labels])
		out = Lambda(lambda x: K.maximum(self.margin-(x[0]-x[1])/x[0], 0))([out_pos, out_neg])
		self._model = Model(inputs+[inp_labels], out)
		self._model.compile(extend_with_piecewise_linear_lr(Adam, name='AdamLR')(
			learning_rate=self.learning_rate, lr_schedule=self.lr_schedule
		), loss=lambda tar,out:out, metrics=[marginal_accuracy(self.margin)])
		if verbose: self._model.summary()
class PairGenerator(DataGenerator):
	def __init__(self, clues, glosses, hits, y_input=False, **kwargs):
		for k, v in kwargs.items(): setattr(self, k, v)
		self.clues = clues
		self.glosses = glosses
		self.hits = hits
		self.epoch_size = sum(len(cs) for w, cs, gs in self.hits)
		self.steps = math.ceil(self.epoch_size / self.batch_size)
		self.y_input = y_input

	def __iter__(self, shuffle=False):
		pairs = []
		for weight, clueids, glossids in self.hits:
			if shuffle:
				gids = np.random.choice(len(glossids), len(clueids)).tolist()
				for i in range(len(clueids)): pairs.append((weight, clueids[i], glossids[gids[i]]))
			else:
				for i in range(len(clueids)): pairs.append((weight, clueids[i], glossids[i%len(glossids)]))

		if not shuffle: 
			np.random.seed(12138)
		np.random.shuffle(pairs)
		#Y_batch = np.arange(self.batch_size)[:,None]
		Y_batch = np.eye(self.batch_size)
		for i in range(0, len(pairs), self.batch_size):
			clues_batch, glosses_batch, weights_batch = [], [], []
			for w, cid, gid in pairs[i:i+self.batch_size]:
				clues_batch.append(self.clues[cid])
				glosses_batch.append(self.glosses[gid])
				weights_batch.append(w)
			pseqs = pad_sequences(clues_batch+glosses_batch, padding="post")
			clues_batch = pseqs[:len(clues_batch)]
			glosses_batch = pseqs[-len(glosses_batch):]
			# clues_batch = pad_sequences(clues_batch, padding="post")
			# glosses_batch = pad_sequences(glosses_batch, padding="post")
			clues_sids = np.zeros_like(clues_batch)
			glosses_sids = np.zeros_like(glosses_batch)
			X_batch = [clues_batch, clues_sids, glosses_batch, glosses_sids]
			if self.y_input:
				yield X_batch + [Y_batch[:len(clues_batch),:len(clues_batch)]], np.ones(len(clues_batch)), np.array(weights_batch)
			else:
				yield X_batch, Y_batch[:len(clues_batch),:len(clues_batch)], np.array(weights_batch)
class PNGenerator(DataGenerator):
	def __init__(self, clues, samples, y_input=False, **kwargs):
		for k, v in kwargs.items(): setattr(self, k, v)
		self.clues = clues
		self.samples = samples
		epoch_size = sum(len(pc) for c, pc, nc in samples)
		self.steps = math.ceil(epoch_size / self.batch_size)
		print(self.steps, "steps")
		self.y_input = y_input
	def __iter__(self, shuffle=False):
		batches = []
		posinds = [pc.copy() for c, pc, nc in self.samples]
		
		updated = True
		batch = []
		if not shuffle: np.random.seed(12138)
		while updated:
			updated = False
			inds = np.arange(len(posinds)).tolist() 
			np.random.shuffle(inds)
			rs = np.random.rand(self.n_negs)
			for j in inds:
				if posinds[j]:
					pi = posinds[j].pop()
					ci, _, nis = self.samples[j]

					if len(nis) != self.n_negs:
						if nis:
							nids = [nis[int(r*len(nis))] for r in rs]
						else:
							nids = [int(r*len(self.clues))for r in rs]
					else: nids = nis

					batch.append((ci, pi, nids))
					updated = True
				if len(batch) == self.batch_size:
					batches.append(batch)
					batch = []
		if batch: batches.append(batch)

		for batch in batches:
			value_batch, labels_batch, key_batch = [], [], []
			for cid, pid, nids in batch:
				key_batch.append(len(value_batch))
				value_batch.append(self.clues[cid])
				labels_batch.append(len(value_batch))
				value_batch.append(self.clues[pid])
				for c in nids:
					value_batch.append(self.clues[c])
			Y_batch = np.zeros((len(key_batch), len(value_batch)))
			Y_batch[np.arange(len(key_batch)), labels_batch] = 1
			#Y_batch[labels_batch, labels_batch] = 1
		
			pseqs = pad_sequences(value_batch, padding="post")
			key_batch = np.array(key_batch)
			value_batch = pseqs.reshape((len(key_batch), self.n_negs+2, pseqs.shape[-1]))
			value_sids = np.zeros_like(value_batch)
			yield [value_batch, value_sids, key_batch], Y_batch#, np.array(labels_batch, dtype="int32")[:,None]
			del key_batch, value_batch, value_sids, labels_batch
class MUGenerator(PNGenerator):
	def __iter__(self, shuffle=False):
		batches = []
		posinds = [pc.copy() for c, pc, nc in self.samples]
		
		updated = True
		batch = []
		if not shuffle: np.random.seed(12138)
		while updated:
			updated = False
			inds = np.arange(len(posinds)).tolist() 
			np.random.shuffle(inds)
			rs = np.random.rand(self.n_negs)
			for j in inds:
				if posinds[j]:
					pi = posinds[j].pop()
					ci, _, nis = self.samples[j]

					if nis and len(nis) != self.n_negs:
						nids = [nis[int(r*len(nis))] for r in rs]
					else: nids = nis

					batch.append((ci, pi, nids))
					updated = True
				if len(batch) == self.batch_size:
					batches.append(batch)
					batch = []
		if batch: batches.append(batch)

		for batch in batches:
			value_batch, labels_batch, key_batch = [], [], []
			negs = [ni for _, _, nids in batch for ni in nids]
			n_negs = len(batch)*self.n_negs
			if len(negs) > n_negs:
				negs = np.random.choice(negs, n_negs).tolist()
			elif len(negs) < n_negs:
				negs += [int(len(self.clues)*r) for r in np.random.rand(n_negs-len(negs))]
			for cid, pid, nids in batch:
				key_batch.append(len(value_batch))
				value_batch.append(self.clues[cid])
				labels_batch.append(len(value_batch))
				value_batch.append(self.clues[pid])
				for _ in range(self.n_negs):
					c = negs.pop()
					value_batch.append(self.clues[c])
			Y_batch = np.zeros((len(key_batch), len(value_batch)))
			Y_batch[np.arange(len(key_batch)), labels_batch] = 1
			#Y_batch[labels_batch, labels_batch] = 1
		
			pseqs = pad_sequences(value_batch, padding="post")
			key_batch = np.array(key_batch)
			value_batch = pseqs.reshape((len(key_batch), self.n_negs+2, pseqs.shape[-1]))
			value_sids = np.zeros_like(value_batch)
			yield [value_batch, value_sids, key_batch], Y_batch#, np.array(labels_batch, dtype="int32")[:,None]
			
			del key_batch, value_batch, value_sids, labels_batch
	
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode',  type=str, default="CLS")
	parser.add_argument('--limit',  type=int, default=0)
	pargs, unk_args = parser.parse_known_args()

	if pargs.mode == "CLS":
		matcher = Siamese(unk_args)
		matcher.train("intermediate/candidates/train.txt", "intermediate/candidates/valid.txt")
	elif pargs.mode == "MGN":
		matcher = Marginal(unk_args)
		matcher.train("intermediate/candidates/train.txt", "intermediate/candidates/valid.txt", limit=pargs.limit)
	elif pargs.mode == "CTR":
		matcher = Contrastive(unk_args)
		matcher.train("data/simclues/cluepairs.txt")
	elif pargs.mode == "CTRMIX":
		matcher = ContrastiveMixup(unk_args)
		matcher.train("data/simclues/cluepairs_bm25.txt")
	elif pargs.mode == "CTRMM":
		matcher = ContrastiveMixupM(unk_args)
		matcher.train("data/simclues/cluepairs_mix.txt")
	elif pargs.mode == "CTRMMM":
		matcher = CTRMMM(unk_args)
		matcher.train("data/simclues/cluepairs_mix123.txt")
	elif pargs.mode == "CTRMGN":
		matcher = ContrastiveMarginal(unk_args)
		matcher.train("data/simclues/cluepairs.txt")
	else: print("Unknown mode:", pargs.mode)
