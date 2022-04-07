# coding=utf-8
import traceback, math, h5py, pickle, keras, json, os, time, argparse, random, csv, six, sys
from collections import OrderedDict, Iterable, defaultdict
import numpy as np
from keras.models import Model
from keras.callbacks import CSVLogger
from keras.preprocessing import sequence
from os.path import exists, join, dirname
from .utils import csv_iter, BaseObject


class BaseModel(BaseObject):
	@property
	def fields_to_save(self): return self.params
	@property
	def params(self): return {
		"n_epochs": 50,
		"batch_size": 64,
	}
	@property
	def options(self): return {
		"select_rules": [("loss", 1)],
		"model_dir": "./models",
		"data_dir": "./data",
		"raw_dir": "./raw",
		"gen_dir": "./generated",
		"h5buf": 0, # whether save the h5 data file
		"dev_ratio": 0.1,
	}
	@property
	def default_model_dir(self):
		return join(self.model_dir, str(self))
	def __init__(self, args=sys.argv[1:], save_dir=None, verbose=False, **kwargs):
		super().__init__(args, verbose=verbose, **kwargs)
		if save_dir:
			self.load(save_dir)
	def eval(self, verbose=True):
		if isinstance(self.valid_data, DataGenerator):
			ret = self._model.evaluate_generator(self.valid_data.__iter__(), steps=len(self.valid_data), verbose=True)
		else:
			ret = self._model.evaluate(*self.valid_data, batch_size=self.batch_size*2, verbose=verbose)
		try: iter(ret)
		except Exception: ret = [ret]
		return {k:v for k, v in zip(self._model.metrics_names, ret)}
	def save(self, save_dir=None):
		if save_dir is None: save_dir = self.default_model_dir
		if not os.path.exists(save_dir): os.makedirs(save_dir)
		if hasattr(self, "_model"):
			self._model.save_weights(os.path.join(save_dir, "model.h5"))
		if hasattr(self, "_tokenizers"):
			for i, t in enumerate(self._tokenizers):
				t.save(os.path.join(save_dir, "feature_%d"%i))
		if hasattr(self, "_classes"):
			for i, c in enumerate(self._classes):
				c.save(os.path.join(save_dir, "class_%d"%i))
		dfile = h5py.File(join(save_dir, "data.h5"), "w")
		save_dict = {}
		for field, value in self.fields_to_save.items():
			value = getattr(self, field, value)
			if type(value) == np.ndarray: dfile[field] = value
			elif type(value) in {list, dict, set} and len(value) > 100: 
				pickle.dump(value, open(join(save_dir, field+".pkl"), "wb"))
			else: save_dict[field] = value
		dfile.close()
		with open(os.path.join(save_dir, "params.json"), "w", encoding="utf-8") as fout:
			json.dump(save_dict, fout, ensure_ascii=False, indent=4)
	def load(self, save_dir=None):
		if save_dir is None: save_dir = self.default_model_dir
		model_props = json.load(open(os.path.join(save_dir, "params.json"), encoding="utf-8"))
		dfile = h5py.File(join(save_dir, "data.h5"), "r")
		for field, value in self.fields_to_save.items():
			if field in dfile: value = dfile[field][:]
			elif exists(join(save_dir, field+".pkl")): 
				value = pickle.load(open(join(save_dir, field+".pkl"), "rb"))
			elif field in model_props: value = model_props[field]
			setattr(self, field, value)
		dfile.close()
		self.build_tokenizers()
		if hasattr(self, "_tokenizers"):
			for i, t in enumerate(self._tokenizers):
				t.load(os.path.join(save_dir, "feature_%d"%i))
		if hasattr(self, "_classes"):
			for i, c in enumerate(self._classes):
				c.load(os.path.join(save_dir, "class_%d"%i))
		self.build_model()
		if hasattr(self, "_model"):
			self._model.load_weights(os.path.join(save_dir, "model.h5"))
	def split_dev(self, data, seed=12138):
		ts, tl, ds, dl = [], [], [], []
		random.seed(seed)
		for sample, label in zip(*data):
			if random.random() < self.dev_ratio:
				ds.append(sample)
				dl.append(label)
			else: 
				ts.append(sample)
				tl.append(label)
		return (ts, tl), (ds, dl)
	def gen_h5_filename(self, filn):
		return join(self.gen_dir, "%s.%s.h5"%(os.path.split(filn)[1], str(self))) if self.h5buf else None
	def train(self, train_files, dev_files=None, save_dir=None, verbose=True, \
		batch_size=None, n_epochs=None, limit=0, **kwargs):
		if save_dir is None: save_dir = self.default_model_dir
		if not os.path.exists(save_dir): os.makedirs(save_dir)
		if n_epochs is None: n_epochs = self.n_epochs
		if batch_size is None: batch_size = self.batch_size

		data = self.load_data(train_files, limit=limit)
		if not hasattr(self, "_tokenizers"): self.build_tokenizers(data)

		if dev_files:
			dev_data = self.load_data(dev_files, limit=limit)
			train_data = data
		else:
			if type(train_files) == list:
				train_files = train_files[0]
			train_data, dev_data = self.split_dev(data)
			dev_files = train_files+".valid"
			train_files = train_files+".train"

		if verbose: print("making dev data")
		valid_data = self.make_data(dev_data, verbose=verbose, h5file=self.gen_h5_filename(dev_files))

		if verbose: print("making training data")
		train_data = self.make_data(train_data, verbose=verbose, h5file=self.gen_h5_filename(train_files), shuffle=True)

		#self.valid_data = valid_data; print(self.eval()); return
		self.fit(train_data, save_dir=save_dir, validation=valid_data, n_epochs=n_epochs, caller=self)
		self.load(save_dir)
	@property
	def class_weight(self): return None
	def fit(self, train_data, save_dir="", validation=None, verbose=True, batch_size=128, \
		n_epochs=50, limit=0, caller=None, **kwargs):
		if caller is None: caller = self
		self.valid_data = validation if validation else train_data
		if not hasattr(self, "_model"): self.build_model()
		if verbose: print('start training model')	
		try:
			apd = False
			logfile = os.path.join(save_dir, 'log.txt')
			if exists(logfile) and os.path.getsize(logfile): 
				inp = input("There is a trained model in %s \n\t1 - quit\n\t2 - continue\
					\n\t3 - retrain\nPlease enter instructions to continue:"%(save_dir))
				if inp.strip() == "1": print("Quit Training"); return
				elif inp.strip() == "2": 
					print("Load model weights and continue training")
					self._model.load_weights(os.path.join(save_dir, "model.h5"))
					apd = True
				else: 
					print("Clear the dir and retrain the model")
			if isinstance(train_data, DataGenerator):
				self._model.fit_generator(train_data.forfit(), epochs=n_epochs, steps_per_epoch=len(train_data), callbacks=[
					LogEvalSave(filename=logfile, caller=caller, earlystopping=n_epochs//4+1, save_dir=save_dir, \
					select_rules=getattr(self, "select_rules", None), append=apd)], class_weight=self.class_weight)
			else:
				self._model.fit(*train_data, epochs=n_epochs, batch_size=batch_size, callbacks=[
					LogEvalSave(filename=logfile, caller=caller, earlystopping=n_epochs//4+1, save_dir=save_dir, \
					select_rules=getattr(self, "select_rules", None), append=apd)], class_weight=self.class_weight)
		except Exception: traceback.print_exc()
		except:	input('Interupt! Press any keys to continue')
	def make_data(self, data, verbose=True, h5file=None, limit=0, shuffle=False):
		if not hasattr(self, "_tokenizers"): self.build_tokenizers(data)
		if h5file and os.path.exists(h5file):
			with h5py.File(h5file, "r") as dfile: 
				i = 0; key = "X_%d"%i; X = []
				while key in dfile:
					X.append(dfile[key][:])
					i += 1
					key = "X_%d"%i
				i = 0; key = "Y_%d"%i; Y = []
				while key in dfile:
					Y.append(dfile[key][:])
					i += 1
					key = "Y_%d"%i
		else:
			X, Y = self.make_training_data(data)
			if type(X) == np.ndarray: X = [X]
			if type(Y) == np.ndarray: Y = [Y]
			if h5file:
				with h5py.File(h5file, "w") as dfile:
					for i, x in enumerate(X):
						dfile.create_dataset("X_%d"%i, data=x)
					for i, y in enumerate(Y):
						dfile.create_dataset("Y_%d"%i, data=y)

		if limit > 0:
			for i in range(len(X)): X[i] = X[i][-limit:]
			for i in range(len(Y)): Y[i] = Y[i][-limit:]

		#if verbose: print([x[:2] for x in X])
		if shuffle:
			for x in X: np.random.seed(1333); np.random.shuffle(x)
			for y in Y: np.random.seed(1333); np.random.shuffle(y)
		return self._data_generator(X, Y)
	def load_data(self, datafile, verbose=False, limit=0):	
		samples = []
		labels = []
		for label, text in csv_iter(datafile):
			labels.append(label)
			samples.append(text)
		return samples[-limit:], labels[-limit:]
	def make_training_data(self, data):
		samples, labels = data
		X = self.make_feature(samples)
		Y = self.make_label(labels)
		return X, Y
	def do_batch(self, samples):
		feats = self.make_feature(samples)
		return self.predict(feats)
	def do(self, sample):
		return self.do_batch([sample])[0]
	def test_with_params(self, test_params, *psargs, **kwargs):
		params = [0]*len(test_params)
		while params and params[0] < len(test_params[0][1]):
			self.test(*psargs, **dict(kwargs, **{k:v[pid] for (k, v), pid in zip(test_params, params)}))
			params[-1] += 1
			for i in range(len(params)-1, 1, -1):
				if params[i] >= len(test_params[i][1]):
					params[i] = 0
					params[i-1] += 1
				else: break
		if test_params:
			if "rec_file" in kwargs:
				with open(kwargs["rec_file"], "a", encoding="utf-8") as fout: fout.write("\n")
		else: self.test(*psargs, **kwargs)
	def test(self, test_file, rec_file=None, rewrite=False, verbose=True, **kwargs):
		banner = str(self)
		for kv in kwargs.items(): banner += ",%s=%s"%kv
		if rec_file:
			recdir = dirname(rec_file) if rec_file else "./"
			if not exists(recdir): os.makedirs(recdir)
			fout = open(join(recdir, "result_%s.txt"%banner), "w", encoding="utf-8")
		samples, labels = self.load_data(test_file)
		preds = self.do_batch(samples, **kwargs)

		res = self.evaluate(preds, labels, samples, fout)

		header = ["Model"] + list(res.keys())
		content = [banner] + ["%.4f"%res[k] for k in res]
		if verbose: 
			print("\t".join(header))
			print("\t".join(content))

		if rec_file:
			if rewrite or not exists(rec_file) or os.path.getsize(rec_file)==0:
				with open(rec_file, "w", encoding="utf-8") as fout:
					fout.write("\t".join(header)+"\n")	
			with open(rec_file, "a", encoding="utf-8") as fout:
				fout.write("\t".join(content)+"\n")
		return res
	def evaluate(self, preds, labels, samples=None, fout=None):
		if samples is None: samples = [""]*len(labels)
		if fout:
			header = ["true/false", "text", "label", "prediction", "TP", "FN", "FP"]
			fout.write("\t".join(header)+"\n")
		TP = PP = RR = acc = 0
		for text, slabel, spred in zip(samples, labels, preds):
			label = tuple(slabel.split(";"))
			pred = tuple(spred.split(";"))

			PP += len(pred)
			RR += len(label)
			t = 0
			while t < len(pred) and t < len(label):
				if pred[t] != label[t]: break
				t += 1
			TP += t
			pos = t == len(pred) == len(label)
			acc += pos
			if fout:
				fout.write("%d\t%s\t%s\t%s\t%s\t%s\t%s\n"%(pos, text, slabel, spred, \
					";".join(label[:t]), ";".join(label[t:]), ";".join(pred[t:])))
		if fout: fout.close()

		if TP:
			prec = TP / PP
			reca = TP / RR
			f1 = 2 * TP / (PP + RR) 
		else: prec = reca = f1 = 0.
		acc /= len(samples)
		return OrderedDict([("accuracy", acc), ("precision", prec), ("recall", reca), ("f1", f1)])
	def run(self):
		print("Model %s"%(str(self)))
		while True:
			text = input("Enter the text > ")
			if text:
				label = self.do(text)
				print(label)
	def predict(self, feats, batch_size=128, **kwargs):
		yy = self._pmodel.predict(feats, batch_size, **kwargs)
		if type(yy) != list: yy = [yy]
		'''
		maybe restore yy
		'''
		return yy
	def make_feature(self, samples, verbose=False):
		return [xx for _tkn in self._tokenizers for xx in _tkn.make(samples)]
	def make_label(self, labels, verbose=False):
		return [yy for _cls in self._classes for yy in _cls.make(labels)]
	def build_tokenizers(self, data=None):
		samples, labels =(None, None) if data is None else data
		self._tokenizers = [Tokenizer(samples)]
		self._classes = [Tokenizer(labels)]
		self._data_generator = lambda x,y:(x,y) # no generator by default
	def build_model(self):
		self._model = self._pmodel = Model(None, None)

def loadList(listfile):
	return [l if l=="\n"else l.strip("\n") for l in open(listfile, encoding="utf-8")]
def saveList(listobj, listfile):
	with open(listfile, "w", encoding="utf-8") as fout:
		for wd in listobj: fout.write(wd if wd=="\n" else wd.strip("\n")+"\n")
class Tokenizer:
	def __init__(self, samples=None, **kwargs):
		self._special_tokens = {
			"padding": "<PAD>",
			"unknown": "<UNK>",
		}
		self.token_lower_freq = 1
		for k, v in kwargs.items(): 
			if k == "special_tokens":
				self._special_tokens = dict(self._special_tokens, **v)
			else:				
				setattr(self, k, v)
		if samples is not None: self.build(samples)
	@property
	def special_tokens(self):
		return self._special_tokens
	@property
	def unk_token_id(self): return self.get(self.special_tokens["unknown"], None)
	def __bool__(self): return hasattr(self, "i_vocab")
	def __len__(self): return len(self.vocab)
	def __iter__(self): return iter(self.vocab)
	def __contains__(self, x): return x in self.i_vocab
	def __getitem__(self, x): return self.vocab[x]
	def token(self, x): return self[x]
	def get(self, x, default=None): return self.i_vocab.get(x, default)
	def token_id(self, x): return self.get(x, self.unk_token_id)
	def load(self, save_file):
		self.vocab = loadList(save_file)
		self.i_vocab = {y:x for x,y in enumerate(self.vocab)}
	def save(self, save_file):
		saveList(self.vocab, save_file)
	def make(self, samples, maxlen=None):
		if not self: self.build(samples)
		feats = []
		for sample in samples:
			feat = self.samp2feat(sample)
			feats.append([self.token_id(f)for f in feat])
		max_len = max(len(l)for l in feats)
		if maxlen is not None: maxlen = min(maxlen, max_len)
		elif max_len == 0: maxlen = 1
		return [sequence.pad_sequences(feats, maxlen=maxlen)]
	def make_iter(self, samples, maxlen=None, batchsize=1):
		if not self: self.build(samples)
		for i in range(0, len(samples), batchsize):
			yield self.make(samples[i:i+batchsize], maxlen=maxlen)
	def recover(self, feats):
		return [[self[f] for f in feat if f] for feat in feats]
	def build(self, samples=None, vocab_file=None, verbose=True):
		if vocab_file and exists(vocab_file):
			id2wd = loadList(vocab_file)
		else:
			special_tokens = [v for v in self.special_tokens.values()]
			special_set = set(special_tokens)
			featcnt = defaultdict(int)
			for sample in samples:
				feat = self.samp2feat(sample)
				for f in feat: featcnt[f] += 1
			id2wd = [w for w, f in sorted(featcnt.items(), key=lambda x:-x[1]) \
				if f >= self.token_lower_freq and w not in special_set]
			if vocab_file: saveList(id2wd, vocab_file)
		if verbose: print("vocab size", len(id2wd)) 
		self.vocab = special_tokens + id2wd
		self.i_vocab = {y:x for x,y in enumerate(self.vocab)}
	def samp2feat(self, sample):
		return sample

class DataGenerator:
	def __init__(self, X, Y=None, batch_size=32, buffer_size=None, **kwargs):
		for k, v in kwargs.items(): setattr(self, k, v)
		self.X = X if type(X) in {list, tuple} else [X]
		self.Y = Y if Y is None or type(Y) in {list, tuple} else [Y]
		self.batch_size = batch_size
		self.steps = (len(self.X[0])-1) // self.batch_size + 1
		self.buffer_size = buffer_size or batch_size * 1000

	def __len__(self):
		return self.steps

	def __iter__(self, shuffle=False):
		inds = np.arange(len(self.X[0])).tolist()
		if shuffle: np.random.shuffle(inds)
		for i in range(0, len(inds), self.batch_size):
			ids = inds[i:i+self.batch_size]
			x_batch = [x[ids] for x in self.X]
			if self.Y is None: yield x_batch
			else: yield x_batch, [y[ids] for y in self.Y]

	def forfit(self):
		while True:
			for d in self.__iter__(True):
				yield d

def handle_value(k):
	is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
	if isinstance(k, six.string_types):
		return k
	elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
		return '"[%s]"' % (', '.join(map(str, k)))
	else:
		return k

class LogEvalSave(CSVLogger):
	def __init__(self, caller, save_dir, select_rules=None, verbose=True, earlystopping=0, save_init=False, **kwargs):
		super().__init__(**kwargs)
		self.caller = caller
		self.verbose = verbose
		self.save_dir = save_dir
		self.select = lambda x:0
		self.earlystopping = earlystopping
		self.best_epoch = 0
		if select_rules:
			self.select = lambda x:tuple(p*x[i]for i, p in select_rules)
			toc = time.time()
			res = self.caller.eval()
			self.rkeys = list(res)
			self.best = self.select(res)
			if self.verbose:
				print('VALIDATION: %s' % ", ".join("%s=%.4f"%kv for kv in res.items()))	
				self.caller.save(self.save_dir)
			if save_init:
				row_dict = OrderedDict({'epoch': 0})
				row_dict.update(("val_%s"%key, res[key]) for key in self.rkeys)
				row_dict.update((("train_time", 0), ("val_time", time.time()-toc)))
				self.init_log = row_dict
			else: self.init_log = None
	def on_epoch_begin(self, *psargs, **kwargs):
		self.tic = time.time()
		super().on_epoch_begin(*psargs, **kwargs)
	def on_epoch_end(self, epoch, logs = (None,)):
		epoch += 1
		logs = logs or {}

		if self.keys is None:
			self.keys = sorted(logs.keys())

		if self.model.stop_training:
			# We set NA so that csv parsers do not fail for this last batch.
			logs = dict([(k, logs[k]) if k in logs else (k, 'NA') for k in self.keys])

		if not self.writer:
			class CustomDialect(csv.excel): delimiter = self.sep
			self.writer = csv.DictWriter(self.csv_file, fieldnames=['epoch'] + \
			["val_%s"%k for k in self.rkeys] + ["train_time", "val_time"] + self.keys, dialect=CustomDialect)
			if self.append_header:
				self.writer.writeheader()
		toc = time.time()
		res = self.caller.eval()
		sel = self.select(res)

		if sel <= self.best:
			self.best = sel
			self.caller.save(self.save_dir)
			self.best_epoch = epoch
		elif self.earlystopping > 0 and epoch > self.earlystopping + max(self.best_epoch, self.earlystopping):
			self.model.stop_training = True
		if self.verbose:
			print('VALIDATION: %s' % ", ".join("%s=%.4f"%kv for kv in res.items()))
		row_dict = OrderedDict({'epoch':epoch})
		row_dict.update(("val_%s"%key, res[key]) for key in self.rkeys)
		row_dict.update((("train_time", toc-self.tic), ("val_time", time.time()-toc)))
		row_dict.update((key, handle_value(logs[key])) for key in self.keys)
		if self.init_log:
			self.writer.writerow(self.init_log)
			self.init_log = None
		self.writer.writerow(row_dict)
		self.csv_file.flush()
