import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict
import stanfordnlp, json, time, io, numpy as np, os, math, sys, random, argparse
from os.path import join

from stanfordnlp.models.tokenize.data import DataLoader
from stanfordnlp.pipeline.doc import Document
from stanfordnlp.models.tokenize.utils import print_sentence
from stanfordnlp.models.common import conll

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.metrics import AUC
from keras.optimizers import Adam
from keras_preprocessing.sequence import pad_sequences

from .puz_utils import string2fill, directions, re_ref
from .utils import BaseObject, csv_iter
from .base import BaseModel, Tokenizer

# other relations: canonical -> C, synonym -> E

'''
TODO: 
0. organize vocab (feats, srcs, etc.)
1. how to distinguish VBD and VBN (or not at all)
2. judge by absolute scores
'''
_class_weight = [ # 1. 1NNS,1n2r, 2. 0.2NNS,0.2n2r, 3. 0.2NNS, 1n1r
	{"n": 1, "v": 1, "a": 1, "r": 2},
	{"NN": 1, "NNS": 1, "VB": 1., "VBZ": 2., "VBG": 5., "VBD": 10., "VBN": 10., \
			"JJ": 1., "JJR": 5., "JJS": 5., "RB": 2., "RBR": 10., "RBS": 10.},
]
nlp = None

word2id, id2word = {}, []
wordinfo, wordtags = [], []
def load_wordforms(dfile="data/dictionaries/word_rels_wikt.txt"):
	with open(dfile, encoding="utf-8") as fin:
		for line in fin:
			word, info = line.strip("\r\n").split("\t", 1)
			word2id[word] = len(id2word)
			id2word.append(word)
			c1, c2 = set(), set()
			wordinfo.append({})
			for k, v in json.loads(info).items():
				wordinfo[-1][k] = v
				c1.add(k)
				if k == "n":
					if "iP" in v: c2.add("NNS")
					else: c2.add("NN")
					if "P" in v: c2.add("NN")
				if k == "v":
					is_trans = False
					if "iZ" in v: 
						c2.add("VBZ")
						is_trans = True
					if "iG" in v: 
						c2.add("VBG")
						c2.add("NN") # gerund can be a noun
						is_trans = True
					if "iD" in v: 
						c2.add("VBD")
						is_trans = True
					if "iN" in v: 
						c2.add("VBN")
						is_trans = True
					if not is_trans or "Z" in v or "G" in v or "D" in v or "N" in v:
						c2.add("VB")
				if k == "a":
					if "iR" in v: c2.add("JJR")
					elif "iS" in v: c2.add("JJS")
					else: c2.add("JJ")
				if k == "r": 
					if "iR" in v: c2.add("RBR")
					elif "iS" in v: c2.add("RBS")
					else: c2.add("RB")
			wordtags.append((c1, c2))
load_wordforms()
def tag_word(word): # the tag of words can be ambiguous
	wid = word2id.get(string2fill(word))
	if wid is None: return set(), set()
	return wordtags[wid]
def tag_words(words): return [tag_word(w) for w in words]
class POSTagger(BaseObject):
	@property
	def initializations(self): return super().initializations + [
		self.init_wpos,
	]
	def init_wpos(self, args, **kwargs):
		self.wpos = []
		self.default_distrib = [np.zeros(len(_cls)) for _cls in self.classes]
		for tags in wordtags:
			distribs = [v.copy() for v in self.default_distrib]
			for i, l in enumerate(tags):
				for k in l:
					distribs[i][self.classes[i].token_id(k)] = 1
			self.wpos.append(distribs)
	@property
	def classes(self): 
		if not hasattr(self, "_classes"):
			self._classes = [
				ClassToken(["n", "v", "a", "r"]),
				ClassToken(["NN", "NNS", "VB", "VBZ", "VBG", "VBD", "VBN", \
					"JJ", "JJR", "JJS", "RB", "RBR", "RBS"]),
			]
		return self._classes
	def tag_clues(self, clues): # the tag of words should be unambiguous
		doc = pipeline(clues)
		
		ret = []
		# if len(doc.sentences) != len(clues):
		# 	for i in range(len(clues)):
		# 		print(clues[i], "\t", [w.text for w in doc.sentences[i].words])
		# 	input()
		for sen in doc.sentences:
			for i, w in enumerate(sen.words):
				if w.dependency_relation == "root":
					if w.xpos.startswith("VB") and i != 0 and i+1!=len(sen.words): pos = "NN"
					else: pos = w.xpos
					if not (pos.startswith("NN") or pos.startswith("VB") or pos == "JJ" or pos == "RB"):
						p = pos = None
					else:
						p = pos[0].lower()
						if p == "j": p = "a"
					ret.append((p, pos, sen.words))
					break
		return ret
	def wpos_distrib(self, word):
		wid = word2id.get(string2fill(word))
		if wid is None: return self.default_distrib
		return self.wpos[wid]
class LSTMTagger(POSTagger, BaseModel):
	@property
	def params(self): return {
		"dropout": 0.2,
		"lstm_dims": 256,
		"emb_dims": "128_64_64",
		"n_epochs": 50,
		"n_layers": 1,
		"tlf": 5,
		"add_cw": 0,
	}
	@property
	def class_weight(self): 
		if self.add_cw:
			ret = []
			for i, _cls in enumerate(self._classes):
				cw = {}
				for k, v in _class_weight[i].items():
					cw[_cls.token_id(k)] = v
				ret.append(cw)
			return ret
		else: return None
	@property
	def options(self): return dict(super().options, **{
		"model_dir": "models/posclassifier",
		"gen_dir": "./intermediate/posclassifier",
		"h5buf": 1,
		"lr": 1e-4,
		"batch_size": 256,
		"dev_ratio": 0.2,
		"pthres": 0.,
	})
	def load(self, save_dir=None): # after loading the models (with vocabulary), re-initialize wpos
		super().load(save_dir=save_dir)
		self.init_wpos(None)
	def build_model(self):
		inps = []
		embs = []
		emb_dims = [int(d) for d in self.emb_dims.split("_")]
		for i in range(len(self._tokenizers)):
			tknz = self._tokenizers[i]
			inp = Input((None,), dtype="int32")
			x = Embedding(len(tknz), emb_dims[i], mask_zero=True)(inp)
			inps.append(inp)
			embs.append(x)
		x = Concatenate()(embs)
		for _ in range(self.n_layers):
			x_fwd = LSTM(self.lstm_dims, return_sequences=True, dropout=self.dropout)(x)
			x_bwd = LSTM(self.lstm_dims, return_sequences=True, dropout=self.dropout, go_backwards=True)(x)
			x = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=-1))([x_fwd, x_bwd])
		x = GlobalAveragePooling1D()(x)
		outps = []
		for c in self._classes:
			outp = Dense(len(c), activation="softmax")(x)
			outps.append(outp)
		self._model = self._pmodel = Model(inps, outps)
		self._model.compile(Adam(self.lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
		self._model.summary()
	def build_tokenizers(self, data=None):
		if data is None:
			samples, labels = [None, None, None], [None, None]
		else:
			samples, labels =  data
		self._tokenizers = [Tokenizer(d, token_lower_freq=self.tlf) for d in samples]
		self._classes = [ClassToken(l) for l in labels]
		self._data_generator = lambda x,y:(x,y) # no generator by default
	def make_label(self, labels, verbose=False):
		return  [yy for i in range(len(self._classes)) for yy in self._classes[i].make(labels[i])]
	def make_feature(self, samples, verbose=False):
		return [xx for i in range(len(self._tokenizers)) for xx in self._tokenizers[i].make(samples[i])]
	def load_data(self, datafile, verbose=False, limit=0):	
		samples = [[], [], []]
		labels = [[], []]
		for label, text, feats in csv_iter(datafile):
			item = json.loads(feats)
			l1, l2 = label.split(",")
			samples[0].append(item['words'])
			samples[1].append(item['postags'])
			samples[2].append(item['deps'])
			labels[0].append(l1)
			labels[1].append(l2)
		print("load_data: preprocessing %d samples"%len(samples[-limit:]))
		return samples, labels
	def split_dev(self, data, seed=12138):
		samples_tra, labels_tra, samples_dev, labels_dev = [], [], [], []
		
		samples, labels = data
		dist = [self.dev_ratio, 1-self.dev_ratio]
		np.random.seed(seed)
		splits = np.random.choice(2, len(samples[0]), p=dist).tolist()

		for ssamples in samples:
			sample_splits = [[], []]
			for i in range(len(ssamples)):
				sample_splits[splits[i]].append(ssamples[i])
			samples_tra.append(sample_splits[1])
			samples_dev.append(sample_splits[0])

		for slabels in labels:
			label_splits = [[], []]
			for i in range(len(slabels)):
				label_splits[splits[i]].append(slabels[i])
			labels_tra.append(label_splits[1])
			labels_dev.append(label_splits[0])
			
		return (samples_tra, labels_tra), (samples_dev, labels_dev)
	def do_batch(self, samples):
		psamples = preproess_samples(samples)
		feats = self.make_feature(psamples)
		preds = self._pmodel.predict(feats, self.batch_size*2)
		labels = [pred.argmax(axis=1) for pred in preds]
		scores = [pred.max(axis=1) for pred in preds]
		return [[(self._classes[i].token(l) if s >= self.pthres else None)
			for l, s in zip(labels[i], scores[i])] for i in range(len(labels))]
	def do_distrib(self, samples):
		psamples = preproess_samples(samples)
		feats = self.make_feature(psamples)
		preds = self._pmodel.predict(feats, self.batch_size*2)
		ret = [[preds[j][i] for j in range(len(self._classes))] for i in range(len(samples))]
		return ret
	def tag_clues(self, clues): # the tag of words should be unambiguous
		ret = []
		pos1, pos2 = self.do_batch(clues)
		for i in range(len(clues)):
			ret.append((pos1[i], pos2[i], None))
		return ret
	def tag_clues_distrib(self, clues): # the tag of words should be unambiguous
		ret = []
		distribs = self.do_distrib(clues)
		for i in range(len(clues)):
			r = []
			for j, _cls in enumerate(self._classes):
				dist = distribs[i][j]
				r.append( [(_cls.token(k), dist[k]) for k in reversed(dist.argsort())] )
			ret.append(r)
		return ret
class ClassToken(Tokenizer):
	def samp2feat(self, sample):
		return [sample] # add 1 dimension

def output_predictions(output_file, trainer, data_generator, vocab, mwt_dict, max_seqlen=1000):
	paragraphs = []
	for i, p in enumerate(data_generator.sentences):
		start = 0 if i == 0 else paragraphs[-1][2]
		length = sum([len(x) for x in p])
		paragraphs += [(i, start, start+length, length+1)] # para idx, start idx, end idx, length

	paragraphs = list(sorted(paragraphs, key=lambda x: x[3], reverse=True))

	all_preds = [None] * len(paragraphs)
	all_raw = [None] * len(paragraphs)

	eval_limit = max(3000, max_seqlen)

	batch_size = trainer.args['batch_size']
	batches = int((len(paragraphs) + batch_size - 1) / batch_size)

	t = 0
	for i in range(batches):
		batchparas = paragraphs[i * batch_size : (i + 1) * batch_size]
		offsets = [x[1] for x in batchparas]
		t += sum([x[3] for x in batchparas])

		batch = data_generator.next(eval_offsets=offsets)
		raw = batch[3]

		N = len(batch[3][0])
		if N <= eval_limit:
			pred = np.argmax(trainer.predict(batch), axis=2)
		else:
			idx = [0] * len(batchparas)
			Ns = [p[3] for p in batchparas]
			pred = [[] for _ in batchparas]
			while True:
				ens = [min(N - idx1, eval_limit) for idx1, N in zip(idx, Ns)]
				en = max(ens)
				batch1 = batch[0][:, :en], batch[1][:, :en], batch[2][:, :en], [x[:en] for x in batch[3]]
				pred1 = np.argmax(trainer.predict(batch1), axis=2)

				for j in range(len(batchparas)):
					sentbreaks = np.where((pred1[j] == 2) + (pred1[j] == 4))[0]
					if len(sentbreaks) <= 0 or idx[j] >= Ns[j] - eval_limit:
						advance = ens[j]
					else:
						advance = np.max(sentbreaks) + 1

					pred[j] += [pred1[j, :advance]]
					idx[j] += advance

				if all([idx1 >= N for idx1, N in zip(idx, Ns)]):
					break
				batch = data_generator.next(eval_offsets=[x+y for x, y in zip(idx, offsets)])

			pred = [np.concatenate(p, 0) for p in pred]

		for j, p in enumerate(batchparas):
			len1 = len([1 for x in raw[j] if x != '<PAD>'])
			if pred[j][len1-1] < 2:
				pred[j][len1-1] = 2
			elif pred[j][len1-1] > 2:
				pred[j][len1-1] = 4
			all_preds[p[0]] = pred[j][:len1]
			all_raw[p[0]] = raw[j]

	offset = 0
	oov_count = 0

	for j in range(len(paragraphs)):
		raw = all_raw[j]
		pred = all_preds[j]

		current_tok = ''
		current_sent = []

		for t, p in zip(raw, pred):
			if t == '<PAD>':
				break
			# hack la_ittb
			if trainer.args['shorthand'] == 'la_ittb' and t in [":", ";"]:
				p = 2
			offset += 1
			if vocab.unit2id(t) == vocab.unit2id('<UNK>'):
				oov_count += 1

			current_tok += t
			if p >= 1:
				tok = vocab.normalize_token(current_tok)
				assert '\t' not in tok, tok
				if len(tok) <= 0:
					current_tok = ''
					continue
				current_sent += [(tok, p)]
				current_tok = ''
				# if p == 2 or p == 4:
				# 	print_sentence(current_sent, output_file, mwt_dict)
				# 	current_sent = []

		if len(current_tok):
			tok = vocab.normalize_token(current_tok)
			assert '\t' not in tok, tok
			if len(tok) > 0:
				current_sent += [(tok, 2)]

		if len(current_sent):
			print_sentence(current_sent, output_file, mwt_dict)

	return oov_count, offset, all_preds
def tokenize(doc):
	global nlp
	if nlp is None: nlp = stanfordnlp.Pipeline()
	processor = nlp.processors["tokenize"]
	config = processor.config
	vocab = processor.vocab

	batches = DataLoader(config, input_text=doc.text, vocab=vocab, evaluation=True)

	with io.StringIO() as conll_output_string:
		output_predictions(conll_output_string, processor.trainer, batches, vocab, None,
			config.get('max_seqlen', 1000))
		doc.conll_file = conll.CoNLLFile(input_str=conll_output_string.getvalue())
def pipeline(texts):
	doc = Document("\n\n".join(texts))
	tokenize(doc)
	for p in ["pos", "lemma", "depparse"]:
		nlp.processors[p].process(doc)
	doc.load_annotations()
	return doc
def preproess_samples(samples, batch_size=1024, verbose=False):
	words, postags, depparses = [], [], []
	n_batches = math.ceil(len(samples)/batch_size)
	t0 = time.time()
	for i in range(n_batches):
		doc = pipeline(samples[i*batch_size:(i+1)*batch_size])
		for sen in doc.sentences:
			ws, ps, ds = [], [], []
			for word in sen.words:
				ws.append(word.text)
				ps.append(word.xpos)
				ds.append(word.dependency_relation)
			words.append(ws)
			postags.append(ps)
			depparses.append(ds)

		if verbose:
			delta_t = time.time()-t0
			sys.stdout.write("\rPipelining: %d/%d, ETA: %.2f"%(i+1, n_batches, delta_t/(i+1)*(n_batches-i-1)))
			sys.stdout.flush()
			if n_batches == i+1: 
				print("Done...%d samples in %.2f secs%s"%(len(samples), delta_t, " "*10))
	return words, postags, depparses
def test(method_args, testfile, outdir="outputs/pos", append=0):
	methods = []
	for ma in method_args:
		ma = ma.split(":")
		mc, kwa = (ma[0], "") if len(ma) == 1 else (ma[0], ma[1])
		margs = dict(kv.split("=", 1) for kv in kwa.split(",")) if kwa else {}
		method = eval(mc)(**margs)
		try:
			method.load()
			print("trained %s loaded"%str(method))
		except Exception: 
			import traceback
			traceback.print_exc()
			print("%s has nothing to load"%(str(method)))
		methods.append(method)

	data = defaultdict(list)
	with open(testfile, encoding="utf-8") as fin:
		for line in fin:
			item = json.loads(line.strip())
			date, pos = item["source"].split(",")
			data[date].append((item["clue"], item["answer"], pos))
	n_clues = sum(len(clues) for clues in data.values())
	print(n_clues, "clue-answer pairs")

	n_clues = 0
	metrics = [{"p1":0, "t1":1e-9, "r1":1e-9, "p2":0, "t2":1e-9, "r2":1e-9, "dt":0.} for m in methods] 
	for date, clues in data.items():
		texts = []
		answs = []
		for clue, ans, pos in clues:
			if "___" not in clue and re_ref.search(clue.lower()) is None:
				texts.append(clue)
				answs.append(ans)
		
		for k in range(len(methods)):
			
			metric = metrics[k]
			method = methods[k]

			t0 = time.time()
			cpos = method.tag_clues(texts)
			apos = tag_words(answs)
			metric["dt"] += time.time()-t0
		
			assert len(cpos) == len(texts) == len(answs) == len(apos)
			# if date == "11/22/2020":
			# 	for c, a in zip(texts, answs):
			# 		print(c, a)
			# 	print("len(cpos)", len(cpos))
			# 	print("len(segs)", len(segs))
			# 	print("len(texts)", len(texts))
			# 	print("len(answs)", len(answs))
			# 	print("len(apos)", len(apos))
		
			for i in range(len(texts)):
				cp1, cp2, cdet = cpos[i]
				ap1, ap2 = apos[i]
				if ap1:
					if cp1 is not None:
						metric["t1"] += 1
						metric["p1"] += cp1 in ap1
					metric["r1"] += 1
				if ap2:
					if cp2 is not None:
						match = cp2 in ap2
						metric["t2"] += 1
						metric["p2"] += match
			
						# if not match:# and answs[i] == "CEDE": 
						# 	print("POS tag mismatch(%s)\t%s -> %s\t%s -> %s"\
						# 	%(date, [(w.text, w.xpos, w.dependency_relation)for w in cdet], \
						# 		answs[i], cp2, ap2))
					metric["r2"] += 1
		n_clues += len(texts)
	if not os.path.exists(outdir): os.makedirs(outdir)
	with open(join(outdir, "res.txt"), "a" if append else "w", encoding="utf-8") as fout:
		head_str = "Method\tL1prec.\tL1reca.\tL1f1\tL2prec.\tL2reca.\tL2f1\ts/puz\tms/clue\n"
		print(head_str, end="")
		if not append:
			fout.write(head_str)
		for k in range(len(metrics)):
			metric = metrics[k]
			p1, t1, r1 = metric["p1"], metric["t1"], metric["r1"]
			p2, t2, r2 = metric["p2"], metric["t2"], metric["r2"]
			acc1_str = "%d/%d=%.2f%%\t%d/%d=%.2f%%\t%.2f%%"%(p1, t1, p1/t1*100, p1, r1, p1/r1*100, 2*p1/(t1+r1)*100)
			acc2_str = "%d/%d=%.2f%%\t%d/%d=%.2f%%\t%.2f%%"%(p2, t2, p2/t2*100, p2, r2, p2/r2*100, 2*p2/(t2+r2)*100)
			tpp_str = "%d/%d=%.2f"%(metric["dt"], len(data), metric["dt"]/len(data))
			tpc_str = "%d/%d=%.1f"%(metric["dt"]*1000, n_clues, metric["dt"]*1000/n_clues)
			res_str = "%s,%s\t%s\t%s\t%s\t%s\n"%(method_args[k], methods[k].banner, acc1_str, acc2_str, tpp_str, tpc_str)
			print(res_str, end="")
			fout.write(res_str)
def make_data(indir, oupath, limit=0, append=0):
	# filter labels
	docs = {}
	for fn in os.listdir(indir):
		with open(join(indir, fn), encoding="utf-8") as fin:
			for line in fin:
				doc, strings = line.rstrip("\r\n").split("\t", 1)
				docs.setdefault(doc.replace("\t"," "), []).extend(strings.split(" "))
	n_clues = len(docs)
	print(n_clues, "clues")
	exs = {}
	if append: # append=1: simply appending the results; append=2: re-tag the words
		with open(oupath, encoding="utf-8") as fin:
			for line in fin:
				label, clue, item = line.strip("\r\n").split("\t")
				exs[clue] = (label, item)
		print("Append mode %d: %d existing clues"%(append, len(exs)))
	bad_labels = {"<MUL>", "<NOS>", "<UNK>"}
	items = []
	n_vals = 0
	pos1cnts = defaultdict(int)
	pos2cnts = defaultdict(int)
	t0 = time.time()
	with open(oupath, "w", encoding="utf-8") as fout:
		while docs:
			clue, words = docs.popitem()
			if "__" in clue or re_ref.search(clue.lower()): continue
			words = [w for w in set(words)if w == string2fill(w)]
			if clue.strip() and words:
				n_vals += 1
				ret = tag_words(words)
				pos1, pos2 = zip(*ret)
				pos1 = [p for p in pos1 if p]
				if pos1: 
					p1 = set.intersection(*pos1)
					if p1:
						if len(p1) == 1: label1 = p1.pop()
						else: label1 = "<MUL>"
					else: label1 = "<NOS>"
				else: label1 = "<UNK>"

				pos2 = [p for p in pos2 if p]
				if pos2: 
					p2 = set.intersection(*pos2)
					if p2:
						if len(p2) == 1: label2 = p2.pop()
						elif p2 == {"VBG", "NN"}: label2 = "VBG"
						else: label2 = "<MUL>"
					else: label2 = "<NOS>"
				else: label2 = "<UNK>"
				
				if label1 not in bad_labels and label2 not in bad_labels:
					pos1cnts[label1] += 1
					pos2cnts[label2] += 1
					if append == 1:
						labels, res = exs.get(clue, (None, None))
						if res is None:
							items.append((label1, label2, clue))
						else:
							fout.write("%s\t%s\t%s\n"%(labels, clue, res))
					elif append == 2:
						labels, res = exs.get(clue, (None, None))
						if res is None:
							items.append((label1, label2, clue))
						else:
							l1, l2 = labels.split(",", 1)
							pos1cnts[l1] -= 1
							pos2cnts[l2] -= 1	
							fout.write("%s,%s\t%s\t%s\n"%(label1, label2, clue, res))
					else:
						items.append((label1, label2, clue))
						
			if (not docs and items) or len(items) >= 10000:
				doc = pipeline([clue for _, _ , clue in items])
				for sen, (label1, label2, clue) in zip(doc.sentences, items):
					ws, ps, ds = [], [], []
					for word in sen.words:
						ws.append(word.text)
						ps.append(word.xpos)
						ds.append(word.dependency_relation)
					fout.write("%s,%s\t%s\t%s\n"%(label1, label2, clue, json.dumps({"words": ws, "postags": ps, "deps":ds})))
				items = []
				delta_t = time.time()-t0
				sys.stdout.write("\rPipelining: %d/%d, ETA: %.2f"%(n_clues-len(docs), n_clues, delta_t/(n_clues-len(docs))*(len(docs))))
				sys.stdout.flush()
				if not docs: 
					print("Done...%d samples in %.2f secs%s"%(n_clues, delta_t, " "*10))
				if limit and limit + len(docs) <= n_clues: break
	print(n_vals, "valid clues")
	print("Counts of pos1:")
	for k, v in sorted(pos1cnts.items(), key=lambda x:-x[1]):
		print("\t", k, v)
	print("Counts of pos2:")
	for k, v in sorted(pos2cnts.items(), key=lambda x:-x[1]):
		print("\t", k, v)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--test_set',  type=str, default="data/clues_for_test.txt")
	parser.add_argument('--train_set',  type=str, default="data/POS/labeled_clues.txt")
	parser.add_argument('--clue_dir',  type=str, default="data/clues")
	parser.add_argument('--test_models',  type=str, \
		default="LSTMTagger;LSTMTagger:pthres=0.5;LSTMTagger:pthres=0.7;LSTMTagger:pthres=0.8;LSTMTagger:pthres=0.9")
	parser.add_argument('--train_model',  type=str, default="LSTMTagger")
	parser.add_argument('--process',  type=str, default="test")
	parser.add_argument('--append',  type=int, default=0)
	pargs, unkargs = parser.parse_known_args()
	process = pargs.process.split(",")

	if "make" in process:
		make_data(pargs.clue_dir, pargs.train_set, append=pargs.append)
	if "train" in process:
		model = eval(pargs.train_model)(unkargs)
		# model.load(model.default_model_dir+",init")
		model.train(pargs.train_set)	
	if "test" in process:
		test(pargs.test_models.split(";"), pargs.test_set, append=pargs.append)
else: poscls = LSTMTagger(pthres=0.7, add_cw=1)