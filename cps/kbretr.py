import os, json, re, string, time, pymongo, numpy as np
from os.path import join, exists
from collections import defaultdict
from datetime import datetime
from elasticsearch import Elasticsearch, helpers
import keras.backend as K
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from .utils import BaseObject
from .puz_utils import tokenize, string2fill, re_short, directions, re_ref, construct_grid
from .wikimapper import get_mapper
from .base import BaseModel
from .wfeats import get_vocab, get_idf


class WikiSplitsES(BaseObject):
	@property
	def options(self): return dict(super().options, **{
		"synset_path": "data/wordnet/synsets.txt",
		"dbname": "wikisplits",
		"src_file_path": "../DPR-master/downloads/data/wikipedia_split/psgs_w100.tsv",
		"new": False,
		"doc_limit": 20,
		"idf_path": "data/idf.txt",
	})
	@property
	def initializations(self): return super().initializations + [
		self.init_es,
	]
	def init_es(self, args, **kwargs):
		self.es = Elasticsearch(**json.load(open("configs/elasticsearch.json")))
		if not self.es.indices.exists(self.dbname) or self.new:
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
			self.construct(self.src_file_path)
	def construct(self, dirn):
		def data_gen(dirn):
			cnt = 0
			t0 = time.time()
			with open(dirn, encoding="utf-8") as fin:
				fin.readline()
				for line in fin:
					_id, remains = line.strip("\r\n").split("\t", 1)
					text, title = remains.rsplit("\t", 1)
					text = text.strip('"').replace('""', '"')
					cnt += 1
					yield {"_index": self.dbname, "_id": _id, "_source": {
						"text": text, 
						"title": title, 
					}}
					if cnt % 10000 == 0: print("%7d\t%.2f"%(cnt, time.time()-t0))
			print("%7d\t%.2f"%(cnt, time.time()-t0))
		helpers.bulk(self.es, (item for item in data_gen(dirn)))
	def retrieve(self, query, limit=None):
		if limit is None: limit = self.doc_limit
		results = self.es.search(index=self.dbname, body={
			"query": {"bool": {
				"must": [
					{"match": {"text": query}}
				],
				"filter": [
				]
				# "must_not": {"term": {"lengths": 10}}
			}},
			"from": 0,
			"size": limit,
		}, request_timeout=100)
		return results["hits"]["hits"]

class KBRetriever(WikiSplitsES):
	@property
	def params(self): return dict(super().params, **{
		"w_page": 1,
		"w_context": 0.5,
		"decay": 0.1,
		"n_props": 1,
		"with_words": 0,
		"with_ents": 1,
		"max_n_words": 1, # max number of words in phrases from contexts
	})
	@property
	def options(self): return dict(super().options, **{
		"dbname": "wikisplits_linked",
		"src_file_path": "/mnt/clh25/wikidata/wiki_splits1.txt",
		"wiki_db_path": "data/wiki/index_enwiki-latest.db",
		"wiki_title_path": "data/wiki/wikititles.txt",
		"pred_cnt_path": "data/wiki/pred_cnts.txt",
		"pred_inv_cnt_path": "data/wiki/inv_pred_cnts.txt",
		"vocab_path": "data/dictionaries/vocab_2020-09-26.txt",
		"limit": 20,
		"n_enames": 1, # limit mentions for one entity
	})
	@property
	def initializations(self): return super().initializations + [
		self.init_kg,
		self.init_idf,
		self.init_stats,
		self.init_params,
	]
	def init_idf(self, args, **kwargs):
		self.idf, self.max_idf = get_idf(path=self.idf_path)
		self._vocabulary = get_vocab(self.vocab_path)
	def init_kg(self, args, **kwargs):
		client = pymongo.MongoClient(**json.load(open("configs/mongodb.json")))
		self.kg_col = client.wikidata.triples
		self.kg_inv_col = client.wikidata.inv_triples
		self.kg_buf = {}
		self.p_weights = defaultdict(lambda:1.)
		with open(self.pred_cnt_path, encoding="utf-8") as fin:
			for line in fin:
				p, avgn, maxn, maxe = line.strip("\r\n").split("\t")
				self.p_weights[p, 0] = 1/float(avgn)
		with open(self.pred_inv_cnt_path, encoding="utf-8") as fin:
			for line in fin:
				p, avgn, maxn, maxe = line.strip("\r\n").split("\t")
				self.p_weights[p, 1] = 1/float(avgn)
		self.blacklist = {"Q30"}
		self.memory_mapper = get_mapper()
	def init_stats(self, args, **kwargs):
		self.timings = {"retr": [0, 0], "prop": [0, 0], "map": [0, 0]}
	def init_params(self, args, **kwargs):
		''' Return features: 
		1. context word score
		2. page entity score
		3. context entity score
		4. prop1 for page entity
		5. prop1 for context entity
		...
		'''
		self.weights = np.zeros(2*self.n_props+3, dtype="float32")
		self.weights[0] = self.with_words * self.w_context
		for i in range(self.n_props+1):
			self.weights[i*2+1] = self.w_page * self.decay**i
			self.weights[i*2+2] = self.w_context * self.decay**i
		print("Cold Start Weights:", self.weights.tolist())
	def construct(self, dirn):
		def data_gen(dirn):
			cnt = 0
			t0 = time.time()
			with open(dirn, encoding="utf-8") as fin:
				for line in fin:
					item = json.loads(line)
					_id = item["sid"]
					title = item["title"]
					wikidata_id = item["wikidata_id"]
					text = item["text"]
					links = "\n".join(
						"%d %d %s"%(offset, length, link)
						for offset, length, _, link in item["links"] if link
					)
					cnt += 1
					yield {"_index": self.dbname, "_id": _id, "_source": {
						"text": text, 
						"title": title,
						"wikidata_id": wikidata_id, 
						"links": links,
					}}
					if cnt % 10000 == 0: print("%7d\t%.2f"%(cnt, time.time()-t0))
			print("%7d\t%.2f"%(cnt, time.time()-t0))
		helpers.bulk(self.es, (item for item in data_gen(dirn)), request_timeout=100)
	def get_page_ents(self, docs, norm=1): 
		ret = defaultdict(float)
		if self.with_ents:
			for item in docs:
				ret[item["_source"]["wikidata_id"]] += item["_score"]/norm
		return ret
	def get_context_ents(self, docs, tokens, norm=1):
		ret = defaultdict(float)
		for item in docs:
			for link in item["_source"]["links"].split("\n"):
				if link.strip():
					offset, l, e = link.split(" ", 2)
					ret[e] += item["_score"]/norm
			if self.with_words:
				tokens = tokenize(item["_source"]["text"])
				for i in range(len(tokens)):
					for j in range(self.max_n_words):
						t = " ".join(tokens[i:i+j+1])
						ft = string2fill(t)
						if len(ft) >= 24: break
						if j > 0 and ft not in self._vocabulary: continue
						ww = self.idf.get(t, self.max_idf)/self.max_idf
						ret[ft] += item["_score"]/norm*ww
		return ret
	def get_triples(self, ent):
		if ent in self.kg_buf: return self.kg_buf[ent]
		else:
			ret = []
			for item in self.kg_col.find({"s":ent}):
				ret.append((item["p"], item["o"], 0))
			for item in self.kg_inv_col.find({"o":ent}):
				ret.append((item["p"], item["s"], 1))
			self.kg_buf[ent] = ret
			return ret
	def is_ent(self, w): return w is not None and bool(re.fullmatch("Q[\d]+", w))
	def propagate(self, graph):
		ngraph = {}
		for e, w in graph.items():
			if self.is_ent(e):
				for p, o, i in self.get_triples(e):
					ngraph[o] = ngraph.get(o, 0) + w*self.p_weights[p, i]
		return ngraph
	def predict(self, samples):
		if not samples: return []
		ents, feats = zip(*samples.items())
		scores = np.dot(feats, self.weights)
		sinds = scores.argsort()
		ret = []
		for i in reversed(sinds):
			ret.append((ents[i], float(scores[i])))
		return ret
	def make_feature_iter(self, clues, verbose=False):
		'''
		filter out words in clue but not length mismatching
		because legnth is random but words in clue may have strong signal
		'''
		for clue in clues:
			wic = set() # words in clue
			norm = 0
			clue = re_short.sub("", clue)
			
			tokens = {}
			for t in tokenize(clue):
				wic.add(string2fill(t))
				idf = self.idf.get(t, self.max_idf)
				norm += idf 
				tokens[t] = idf
			for t in tokens: tokens[t] /= sum(tokens.values())

			t0 = time.time()
			docs = self.retrieve(clue)
			pents = self.get_page_ents(docs)
			if verbose: print(len(set(pents)), "page entities")
			cents = self.get_context_ents(docs, tokens)
			if verbose: print(len(set(cents)), "context entities")
			self.timings["retr"][0] += time.time()-t0
			self.timings["retr"][1] += 1

			
			t0 = time.time()
			cwords = {}
			for e in list(cents):
				if not self.is_ent(e):
					cwords[e] = cents.pop(e)
			graphs = [cwords, pents, cents]
			for i in range(self.n_props):
				graphs.append(self.propagate(graphs[-2]))
				graphs.append(self.propagate(graphs[-2]))
			if verbose:
				pes = set.union(*[set(g) for g in graphs[3:]])
				print(len(pes), "propagated entities")
			self.timings["prop"][0] += time.time()-t0
			self.timings["prop"][1] += 1

			feats = {}
			for i, ents in enumerate(graphs):
				for e, w in ents.items():
					if e not in feats: feats[e] = np.zeros_like(self.weights)
					feats[e][i] = w
		
			# filter out words in clue
			for e in list(feats):
				if e in wic:
					feats.pop(e)
			yield feats
	def make_feature(self, clues, verbose=False):
		return list(self.make_feature_iter(clues, verbose))
	def generate(self, clue, length, limit=None, blacklist=set(), verbose=False):
		if limit is None: limit = self.limit
		feats = self.make_feature([clue], verbose=verbose)[0]
		scores = self.predict(feats)
		
		exs = blacklist.copy()
		for t in tokenize(clue):
			exs.add(string2fill(t))

		ret = []
		t0 = time.time()
		for e, w in scores:
			if self.is_ent(e): elist = self.memory_mapper.id_to_titles(e)
			elif e: elist = [string2fill(e)]
			else: elist = []
			nm = 0
			for s in elist:
				if len(s) == length and s not in exs:
					ret.append((s, w, e))
					exs.add(s)
					nm += 1
					if self.n_props > 0 and nm >= self.n_enames: break
			if len(ret) >= limit: break
		self.timings["map"][0] += time.time()-t0
		self.timings["map"][1] += 1
		return ret
class KBR_DW(KBRetriever): # with distance weights
	def get_context_ents(self, docs, tokens, norm=1):
		ret = defaultdict(float)
		for item in docs:
			text = item["_source"]["text"].lower()
			char2pos = []
			pos = 0
			lst_ws = False
			for c in text:
				if c in string.whitespace:
					if not lst_ws: pos += 1
					char2pos.append(None)
					lst_ws = True
				else:
					char2pos.append(pos)
					lst_ws = False
			char2pos.append(pos+1)

			posw = []
			start = 0
			for i in range(len(text)+1):
				if i == len(text) or text[i] in string.whitespace:
					if start < i:
						st = text[start:i].strip().strip(string.punctuation)
						if st in tokens:
							posw.append((char2pos[start], tokens[st]))
					start = i+1

			if self.with_ents:
				for link in item["_source"]["links"].split("\n"):
					if link.strip():
						offset, l, e = link.split(" ", 2)
						end = int(offset)
						offset = end - int(l)

						# print("%s[%s]%s\t->\t%s"%(
						# 	text[offset-20:offset],
						# 	text[offset :end],
						# 	text[end:end+20],
						# 	self.memory_mapper.id_to_titles(e)[:2]
						# ))
						e_offset = None
						while e_offset is None:
							e_offset = char2pos[offset]
							if e_offset is None and end <= len(text): e_offset = char2pos[end-1]
							if offset > 0: offset -= 1
							if end < len(text): end += 1

						pw = 0
						for w_offset, w in posw:
							if w_offset != e_offset: # exclue itself
								pw += w / (1+abs(w_offset-e_offset))
						ret[e] += pw*item["_score"]/norm

			if self.with_words:
				starts = [0]
				for i in range(len(text)+1):
					if i == len(text) or text[i] in string.whitespace:
						if starts[-1] != i:
							for j in range(min(len(starts), self.max_n_words)):
								start = starts[-j-1]
								token = text[start:i]
								ff = string2fill(token)
								if len(ff) >= 24: break
								if 2 < len(ff):
									if j > 0 and ff not in self._vocabulary: continue
									ww = self.idf.get(token, self.max_idf)/self.max_idf
									pw = 0
									for w_offset, w in posw:
										pw += w/(1+abs(w_offset - char2pos[start]))
									ret[ff] += item["_score"]/norm*ww
							starts.append(i+1)
						else: starts[-1] += 1
		return ret
class KBRM(KBRetriever, BaseModel):
	@property
	def params(self): return dict(super().params, **{
		"n_negs": 1023, # batch_size: 1024
	})
	@property
	def options(self): return dict(super().options, **{
		"puz_start_date": "3/25/2019",
		"model_dir": "./models/kbretr",
		"gen_dir": "./intermediate/kbretr",
	})
	def init_params(self, args, **kwargs):
		print("Setting cold start weights")
		super().init_params(args, **kwargs)
		try: 
			self.load()
			print("Setting trained weights")
			self.weights = self._pmodel.get_weights()[0].flatten()
			print("Trained Weights:", self.weights.tolist())
		except Exception: 
			import traceback
			traceback.print_exc()
			
	def build_model(self):
		in_feats = Input((len(self.weights), ), dtype="float32")
		linear_layer = Dense(1)
		out_s = linear_layer(in_feats)
		linear_layer.set_weights([self.weights[:, None], np.zeros(1)])
		self._pmodel = Model(in_feats, out_s)

		in_feats = Input((self.n_negs+1, len(self.weights), ), dtype="float32")
		out_scores = linear_layer(in_feats)
		out_probs = Lambda(lambda x:K.softmax(K.squeeze(x, -1)))(out_scores)
		self._model = Model(in_feats, out_probs)
		self._model.compile(Adam(1e-3), "sparse_categorical_crossentropy", ["accuracy"])
	def build_tokenizers(self, data=None):
		samples, labels =(None, None) if data is None else data
		self._tokenizers = []
		self._classes = []
		self._data_generator = lambda x,y:(x,y) # no generator by default
	def make_training_data(self, data):
		samples, answers = data
		X, Y = [], []
		hits = n_samples = 0
		for i, feats in enumerate(self.make_feature_iter(samples)):
			if not feats: continue
			ents, fvecs = zip(*feats.items())
			pos, neg = [], []
			for j in range(len(ents)):
				if self.is_ent(ents[j]): 
					label = int(answers[i] in self.memory_mapper.id_to_titles(ents[j]))
				else: label = int(answers[i] == ents[j])
				(pos if label else neg).append(fvecs[j])
			if pos:
				neg = np.array(neg)
				for pv in pos:
					x = np.zeros((self.n_negs+1, len(pv)), dtype="float32")
					sinds = np.random.choice(len(neg), self.n_negs, True)
					x[1:] = neg[sinds]
					x[0] = pv
					X.append(x)
				hits += 1
			n_samples += len(feats)
		X = np.array(X)
		print("Recall: %d/%d=%.2f%%"%(hits, len(samples), hits/len(samples)))
		print("Average samples: %.1f"%(n_samples/len(samples)))
		print(X.shape)
		return X, np.zeros((len(X), 1))
	def load_data(self, datafile, verbose=False):
		samples, labels = [], []
		sd = datetime.strptime(self.puz_start_date, "%m/%d/%Y")
		with open(datafile, encoding="utf-8") as fin:
			for line in fin:
				pzl = json.loads(line.strip())
				pt = datetime.strptime(pzl["date"], "%m/%d/%Y")
				if pt >= sd:
					try:
						grid, num2pos, pos2num, clues, answers, agrid = construct_grid(pzl)
					except Exception: continue
					for i in range(len(clues)):
						for num, (clue, length) in clues[i].items():
							for _ in range(5):
								ref = re_ref.search(clue.lower())
								if not ref: break
								s, t = ref.span()
								dnum, d = ref.groups()
								clue = "%s \"%s\" %s"%(clue[:s], clues[directions.index(d)][int(dnum)], clue[t:])
							samples.append(clue)
							labels.append(answers[i][num])
		return samples, labels
class KBRM_DW(KBR_DW, KBRM, BaseModel):	pass					
							
def test_wikies_eff():
	pzlfile = "data/puzzles/nyt.new.ra.txt"
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
	wikidb = WikiSplitsES()
	n_ents = tot = 0
	t0 = time.time()
	for clue, ans, pzl in data:
		n_ents += len(set(item["_source"]["title"] for item in wikidb.retrieve(clue)))
		tot += 1
		if tot % 100 == 0 or tot == len(data): print("%d\t%.2f"%(tot, time.time()-t0))
	print("average entity num", n_ents/tot)

if __name__ == "__main__":
	# test_wikies_eff()

	# cluedb = WikiSplitsES()
	# t0 = time.time()
	# ret = cluedb.retrieve('The Hokies of the A.C.C.')
	# delta_t = time.time()-t0
	# for item in ret:
	# 	item["_source"]["text"] = len(item["_source"]["text"])
	# 	print(item)
	# print(len(ret), delta_t)

	kbretr = KBRetriever(n_props=1)
	t0 = time.time()
	clue, length = ('The Hokies of the A.C.C., for short', 6)#("Fancy open faced sandwich", 7)#
	ret = kbretr.generate(clue, length, 50, verbose=True)
	#ret = kbretr.retrieve(clue)
	delta_t = time.time()-t0
	#for x in ret: print(x)
	for r in ret:
		print(r)
	print(delta_t)
	print(kbretr.timings)