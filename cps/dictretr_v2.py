''' Dictionary Retriever: retrieving words by the glosses from dictionary (wordnet)
	1. w2v retriever: retrieve by w2v
	2. BERT retriever: retrieve by BERT encoding
	3. FT: fine-tuning bert with contrastive learning of clues/glosses
'''
import os, json, string, numpy as np, h5py, time, argparse, re, string, math, sys
from os.path import join, exists
from collections import defaultdict
from elasticsearch import Elasticsearch, helpers
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from .base import BaseModel, DataGenerator
from .puz_utils import string2fill
from .utils import BaseObject
from .wfeats import get_w2v
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.losses import sparse_categorical_crossentropy
from keras_preprocessing.sequence import pad_sequences
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import to_array
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.models import build_transformer_model

class DictRetriever(BaseObject):
	@property
	def options(self): return dict(super().options, **{
		"limit": 10,
		"dbname": "dict_wn", # "dict_wn_ub"
		"dict_path": "data/clue2gloss/glosses.txt", # "data/clue2gloss_ub/glosses.txt"
		"pagesize": 20,
		"es_host": "10.176.64.111:9200",
	})
	@property
	def initializations(self): return [
		self.init_es,
		self.load_glosses,
	]
	def load_glosses(self, args, **kwargs):
		self.glosses = []
		self.words = []
		self.wordset = defaultdict(list)

		with open(self.dict_path, encoding="utf-8") as fin:
			for line in fin:
				gloss, words = line.strip("\r\n").rsplit("\t", 1)
				fwords = list({self.fit2grid(w) for w in words.split(" ")})
				for w in fwords:
					self.wordset[w].append(len(self.glosses))
				self.glosses.append(gloss)
				self.words.append(fwords)

		print(len(self.glosses), "glosses")
		print(len(self.wordset), "grid words")
	def init_es(self, args, new=False, **kwargs):
		self.es = Elasticsearch(self.es_host)
		if not self.es.indices.exists(self.dbname) or new:
			print("index doesn't exists, creating one")
			mapping = {
				'properties': {
					'text': {
						'type': 'text',
					}
				}
			}
			print("Clearing the old db...")
			print(self.es.indices.delete(index=self.dbname, ignore=[400, 404]))
			print("Creating the new db...")
			print(self.es.indices.create(index=self.dbname, ignore=400))
			print("Specifying the index...")
			result = self.es.indices.put_mapping(index=self.dbname, body=mapping)
			print(result)
			self.construct(self.dict_path)
		
	def construct(self, data_file):
		docs = {}
		with open(data_file, encoding="utf-8") as fin:
			for line in fin:
				gloss, words = line.strip("\r\n").rsplit("\t", 1)
				docs.setdefault(gloss, []).extend(words.split(" "))
					
		print(len(docs), "unique glosses")
		def data_gen(docs):
			tot = cnt = 0
			t0 = time.time()
			while docs:
				doc, strings = docs.popitem()
				strings = list({string2fill(s) for s in strings})
				tot += len(strings)
				lens = {len(s)for s in strings}
				cnt += 1
				yield {"_index": self.dbname, "_source": {
					"text": doc, 
					"words": " ".join(strings), 
					"lengths": sorted(list(lens))
				}}
				if cnt % 1000 == 0 or not docs: print("%7d\t%.2f"%(cnt, time.time()-t0))
			print(tot, "gloss-word pairs")
		helpers.bulk(self.es, (item for item in data_gen(docs)), request_timeout=100)
	def fit2grid(self, ss): return string2fill(ss)
	def generate(self, clue, length, limit=None, blacklist=set()):
		if limit is None: limit = self.limit
		ret = []
		exs = blacklist.copy()
		pfrom = 0
		while len(ret) < limit is not None:
			results = self.es.search(index=self.dbname, body={
				"query": {"bool": {
					"must": [
						{"match": {"text": clue}}
					],
					"filter": [
						{"term": {"lengths": length}},
					]
					# "must_not": {"term": {"lengths": 10}}
				}},
				"from": pfrom,
				"size": self.pagesize,
			})
			pfrom += self.pagesize
			hits = results["hits"]["hits"]
			for item in hits:
				ws = set(s for s in item["_source"]["words"].split(" ") if len(s) == length and s not in exs)
				if ws:
					for w in ws:
						ret.append((w, item["_score"], item["_source"]["text"]))
					exs |= ws
				if len(exs) >= limit: break
			if len(hits) < self.pagesize: break
		return ret
	def generate_batch(self, clues, lengths, limit=None, blacklist=set()):
		ret = []
		for clue, length in zip(clues, lengths):
			ret.append(self.generate(clue, length, limit=limit, blacklist=blacklist))
		return ret
	def test_one(self, ans, clue, limit=50, blacklist=set(), fdet=None):
		res = self.generate(clue, len(ans), limit, blacklist)
		self.eval_one(res, ans, clue, fdet)
	def eval_one(self, res, ans, clue, fdet):
		rank = self.update_metric(res, ans)
		if fdet is not None:
			gt = [self.glosses[i_syn] for i_syn in self.wordset[self.fit2grid(ans)]]
			fdet.write(f"\n{clue}\t{ans}\t{rank}\t{'HIT' if rank <= len(res) else gt}\n")
			for w, s, g in res:
				fdet.write("%s\t%.3f\t%s\n"%(w, s, g))
				if w == ans: break
			fdet.flush()
	def update_metric(self, res, ans):
		rank = len(self.wordset)
		for i in range(len(res)):
			k, v, c = res[i]
			if k.lower() == ans.lower(): 
				rank = i
				break
		for j in self.acc:
			if j > rank: self.acc[j] += 1
		self.n += 1
		self.srank += rank+1
		if rank < len(res): self.rrank += 1/(rank+1)
		return rank+1
	def	clear_metric(self): 
		self.acc = {10:0, 20:0, 50:0, 100:0}
		self.n = 0
		self.srank = 0
		self.rrank = 1
	def print_metrics(self):
		return "MRR\t%s"%("\t".join("hits@%d"%k for k in self.acc))
	def print_results(self):
		return "%.4f\t%s" % (self.rrank/self.n, "\t".join("%2.2f"%(self.acc[k]/self.n*100) for k in self.acc))
class W2VRetriever(DictRetriever, BaseModel):
	@property
	def params(self): return dict(super(DictRetriever, self).params, **{
		"_vectors": None,
	})
	@property
	def options(self): return dict(super().options, **{
		"w2v_path": "data/w2v_unorm/",
		"model_dir": "models/dictretrs",
		"norm": 1,
		"thres": 0.
	})
	@property
	def initializations(self): return [
		self.load_model,
		self.load_glosses,
	]
	def load_model(self, args, **kwargs):
		self.wd2id, self.wvecs = get_w2v(self.w2v_path)
	@property
	def vectors(self):
		if not hasattr(self, "_vectors") or self._vectors is None:
			self._vectors = self.text2vec_batch(self.glosses)
			if self.norm:
				self._vectors /= np.linalg.norm(self.vectors, axis=1)[:, None]+1e-9
		return self._vectors
	def build_tokenizers(self, *psargs, **kwargs): 
		self._tokenizers = []
		self._classes = []
	def build_model(self):
		self._model = None
	def text2vec(self, text): return self.text2vec_batch([text])[0]
	def tokenize(self, text):
		wlist = [self.wd2id.get(w.strip().strip(string.punctuation).lower())for w in text.split()]
		wlist = [w for w in wlist if w]
		return wlist
	def text2vec_batch(self, data, verbose=False):
		ret = np.zeros((len(data), self.wvecs.shape[1]))
		for i, text in enumerate(data):
			wlist = self.tokenize(text)
			if wlist:
				ret[i] = self.wvecs[wlist].sum(axis=0)/len(wlist)
		return ret
	def calc_scores(self, vec):
		exp_flag = False
		if len(vec.shape) != 2:
			vec = vec[None, :]
			exp_flag = True
		scores = vec.dot(self.vectors.T)
		if self.norm: scores /= np.linalg.norm(vec, axis=1)[:, None]+1e-9
		if exp_flag: scores = scores[0]
		return scores
	def generate(self, clue, length, limit=50, blacklist=set()):
		vec = self.text2vec(clue)
		scores = self.calc_scores(vec)
		isnan = np.isnan(scores)
		scores[isnan] = 0
		inds = (-scores).argsort().tolist()
		ret = []; exs = blacklist.copy()
		for i in inds:
			s = float(scores[i])
			if s < self.thres: break
			for w in self.words[i]:
				if len(w) == length and w not in exs:
					ret.append((w, s, self.glosses[i]))
					exs.add(w)
			if len(ret) >= limit: break
		return ret
	def generate_batch(self, clues, lengths, limit=50, blacklist=set()):
		vecs = self.text2vec_batch(clues)
		scores = self.calc_scores(vecs)
		inds = (-scores).argsort(axis=-1)
		ret = []
		for k, length in enumerate(lengths):
			ret_k = []; exs = blacklist.copy()
			for i in inds[k]:
				for w in self.words[i]:
					if len(w) == length and w not in exs:
						ret_k.append((w, float(scores[k][i]), self.glosses[i]))
						exs.add(w)
				if len(ret_k) >= limit: break
			ret.append(ret_k)
		return ret
class W2VNorm(W2VRetriever):
	@property
	def options(self): return dict(super().options, **{
		"w2v_path": "data/w2v/",
	})
class BERTRetriever(W2VRetriever):
	def build_model(self, verbose=True):
		bert = build_transformer_model(
			config_path=join(self.bert_path, "bert_config.json"), 
			checkpoint_path=join(self.bert_path, "bert_model.ckpt"), 
			return_keras_model=False,
		)
		cls_emb = Lambda(lambda x:x[:, 0])(bert.model.output)
		self._encoder = Model(bert.model.input, cls_emb)
		if verbose: self._encoder.summary()
	def text2vec_batch(self, data, verbose=False, batch_size=256):
		X = []; preds = []
		t0 = time.time()
		for i in range(len(data)):
			X.append(self._tknz.encode(data[i]))
			if len(X) == batch_size or X and i+1 == len(data):
				X = [pad_sequences(x, padding="post", maxlen=self.maxlen) for x in zip(*X)]
				preds.append(self._encoder.predict(X, batch_size=batch_size))
				X = []
				sys.stdout.write("%d/%d, ETA: %ds\r"%(i+1, len(data), (time.time()-t0)/(i+1)*(len(data)-i-1)))
				sys.stdout.flush()
		# print("BERT prediction done, %.2f secs"%(time.time()-t0))
		preds = np.concatenate(preds, axis=0)
		return preds
	@property
	def options(self): return dict(super().options, **{
		"bert_path": "../bert/wwm_uncased_L-24_H-1024_A-16",
		"maxlen": 100,
		"norm": 0,
	})
	def load_model(self, args, **kwargs):
		self._tknz = Tokenizer(join(self.bert_path, "vocab.txt"), do_lower_case="uncased" in self.bert_path)
		self.build_model()
class BERTRetrieverFT(BERTRetriever):
	@property
	def params(self): return dict(super().params, **{
		"batch_size": 64,
		"fix_layers": 4,
		"fix_embs": 1,
	})
	@property
	def options(self): return dict(super().options, **{
		"bert_path": "../bert/uncased_L-8_H-512_A-8",
		"maxlen": 50,
	})
	def build_model(self, verbose=True):
		# bert1 = build_transformer_model(
		# 	config_path=join(self.bert_path, "bert_config.json"), 
		# 	checkpoint_path=join(self.bert_path, "bert_model.ckpt"), 
		# 	return_keras_model=False,
		# )
		# bert2 = build_transformer_model(
		# 	config_path=join(self.bert_path, "bert_config.json"), 
		# 	checkpoint_path=join(self.bert_path, "bert_model.ckpt"), 
		# 	return_keras_model=False,
		# )
		# inputs = bert1.model.input + bert2.model.input
		# for layer in bert2.model.layers:
		# 	if hasattr(layer, "name"):
		# 		layer.name += "-2"
		# 	else: print(layer, dir(layer))
		# key_encod = Lambda(lambda x:x[:, 0])(bert1.model.output)
		# value_encod = Lambda(lambda x:x[:, 0])(bert2.model.output)

		super().build_model(verbose=False)
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

		encod_dot = Lambda(lambda x:K.dot(x[0], K.transpose(x[1])))([key_encod, value_encod])
		out_prob = Softmax(axis=-1)(encod_dot)
		self._model = Model(inputs, out_prob)
		self._model.compile(Adam(1e-4), "categorical_crossentropy", ["accuracy"])
		if verbose: self._model.summary()
	def load_data(self, datadir, limit=0):
		clues = []
		with open(join(datadir, "clues.txt"), encoding="utf-8") as fin:
			for line in fin:
				clue, _ = line.strip("\r\n").rsplit("\t", 1)
				clues.append(clue)
		glosses = []
		with open(join(datadir, "glosses.txt"), encoding="utf-8") as fin:
			for line in fin:
				gloss, _ = line.strip("\r\n").rsplit("\t", 1)
				glosses.append(gloss)
		hits = []
		with open(join(datadir, "wordhits.txt"), encoding="utf-8") as fin:
			for line in fin:
				_, clueids, glossids = line.strip("\r\n").split("\t")
				hits.append([eval(clueids), eval(glossids)])
				#if len(hits) >= 1000: break
		return (clues, glosses), hits
	def split_dev(self, data, seed=12138):
		(clues, glosses), hits = data
		tl, dl = [], []
		for clues_and_glosses in hits:
			if np.random.random() < self.dev_ratio:
				dl.append(clues_and_glosses)
			else: 
				tl.append(clues_and_glosses)
		return ((clues, glosses), tl), ((clues, glosses), dl)
	def make_data(self, data, verbose=True, h5file=None, limit=0, shuffle=False):
		(clues, glosses), hits = data
		vclues, vglosses = set(), set()
		for hclues, hglosses in hits:
			for c in hclues: vclues.add(c)
			for g in hglosses: vglosses.add(g)
		for i in vclues:
			if type(clues[i]) == str: clues[i], _ = self._tknz.encode(clues[i], maxlen=self.maxlen)
		for i in vglosses:
			if type(glosses[i]) == str: glosses[i], _ = self._tknz.encode(glosses[i], maxlen=self.maxlen)
		return PairGenerator(clues, glosses, hits, batch_size=self.batch_size)
class PairGenerator(DataGenerator):
	def __init__(self, clues, glosses, hits, **kwargs):
		for k, v in kwargs.items(): setattr(self, k, v)
		self.clues = clues
		self.glosses = glosses
		self.hits = hits
		self.epoch_size = sum(len(cs) for cs, gs in self.hits)
		self.steps = math.ceil(self.epoch_size / self.batch_size)

	def __iter__(self, shuffle=False):
		pairs = []
		for clueids, glossids in self.hits:
			if shuffle:
				gids = np.random.choice(len(glossids), len(clueids)).tolist()
				for i in range(len(clueids)): pairs.append((clueids[i], glossids[gids[i]]))
			else:
				for i in range(len(clueids)): pairs.append((clueids[i], glossids[i%len(glossids)]))

		if not shuffle: 
			np.random.seed(12138)
		np.random.shuffle(pairs)
		#Y_batch = np.arange(self.batch_size)[:,None]
		Y_batch = np.eye(self.batch_size)
		for i in range(0, len(pairs), self.batch_size):
			clues_batch, glosses_batch = [], []
			for cid, gid in pairs[i:i+self.batch_size]:
				clues_batch.append(self.clues[cid])
				glosses_batch.append(self.glosses[gid])
			pseqs = pad_sequences(clues_batch+glosses_batch, padding="post")
			clues_batch = pseqs[:len(clues_batch)]
			glosses_batch = pseqs[-len(glosses_batch):]
			# clues_batch = pad_sequences(clues_batch, padding="post")
			# glosses_batch = pad_sequences(glosses_batch, padding="post")
			clues_sids = np.zeros_like(clues_batch)
			glosses_sids = np.zeros_like(glosses_batch)
			yield [clues_batch, clues_sids, glosses_batch, glosses_sids], Y_batch[:len(clues_batch),:len(clues_batch)]

def test_rank(pzlfile, odir, limit=100):
	if not exists(odir): os.makedirs(odir)
	fn = os.path.split(pzlfile)[-1]
	data = []
	with open(pzlfile, encoding="utf-8") as fin:
		for i, line in enumerate(fin):
			pzl = json.loads(line.strip())
			for direction in ["down", "across"]:
				clues = pzl["clues"][direction]
				answs = pzl["answers"][direction]
				for clue, ans in zip(clues, answs):
					data.append((clue, ans, pzl["date"]))
	print(len(data), "clue-answer pairs")
	methods = [DictRetriever(), W2VRetriever(norm=1), BERTRetrieverFT(norm=1)]
	for m in methods:
		try:
			m.load()
			print("%s loaded"%str(m))
		except Exception:
			print("Loading %s failed"%str(m))
		if hasattr(m, "_vectors") and m._vectors is None:
			print("Making vectors of dictionary...")
			m.vectors
			print("Done. saving...")
			m.save()
	for m in methods: m.clear_metric()

	data = [(clue, ans, date) for clue, ans, date in data if ans in methods[0].wordset]
	print(len(data), "golden clue-answer pairs")

	detouts = {str(m):open(join(odir, fn+".%s"%str(m)), "w", encoding="utf-8") for m in methods}
	lst_date, t0 = "", time.time()
	with open(join(odir, fn), "w", encoding="utf-8") as fout:
		fout.write("Method\t%s\n"%methods[0].print_metrics())
		bufs = []
		for i, (clue, ans, date) in enumerate(data):
			index, clue = clue.split(". ", 1)
			index = int(index)
			bufs.append((clue, ans, date))
			
			if (i+1)%100 == 0 or i+1 == len(data): 
				clues, anss, dates = zip(*bufs)
				lengths = [len(a) for a in anss]
				bufs = []
				for method in methods:
					ret = method.generate_batch(clues, lengths, limit)
					for j in range(len(ret)):
						method.eval_one(ret[j], anss[j], clues[j], detouts[str(method)])

				print("%d\ttotal: %7.2fs"%(i+1, time.time()-t0))
				for method in methods:
					rstr = "%s\t%s\n"%(method, method.print_results())
					print(rstr, end="")
					if i+1 == len(data):
						fout.write(rstr)
	for fp in detouts.values(): fp.close()
if __name__ == "__main__":
	# wr = BERTRetriever1()
	# ret, det = wr.generate("Pink or close to it", 4, 20, return_details=True)
	# for d in det: print(d)
	# wr.save()
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode',  type=str, default="test")
	parser.add_argument('--testset',  type=str, default="data/puzzles/nyt.new.ra.txt")
	parser.add_argument('--trainset',  type=str, default="data/clue2gloss")
	parser.add_argument('--outdir',  type=str, default="outputs/answer_rank_dictretr_new")
	pargs, unkargs = parser.parse_known_args()
	if pargs.mode == "test":
		test_rank(pargs.testset, pargs.outdir)
	elif pargs.mode == "train":
		model = BERTRetrieverFT(unkargs)
		model.train(pargs.trainset, n_epochs=20)
