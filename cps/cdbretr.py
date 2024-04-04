import os, json, re, string, time, numpy as np
from elasticsearch import Elasticsearch, helpers
from os.path import join, exists
from .puz_utils import tokenize, string2fill, construct_grid
from .wfeats import get_idf

special_tokens = {
	"___": "SPECIALTOKENBLANK"
}
class ClueES:
	''' 
	Seen clue retriever with only BM25 score
	'''
	def __init__(self, dbname="cluedblfs", dirn="data/clues_before_2020-09-26/", \
	new=False, limit=50, norm=0, base=-1, upper=0, idf_path="data/idf.txt", **kwargs):
		print("ClueES", dbname)
		self.norm = norm
		self.base = base
		self.upper = upper
		self.limit = limit
		self.dbname = dbname
		self.es = Elasticsearch(**json.load(open("configs/elasticsearch.json")))
		if not self.es.indices.exists(self.dbname) or new:
			print("index doesn't exists, creating one")
			self.initialize()
			self.construct(dirn)

		# load idf
		self.idf, self.max_idf = get_idf()
		
		self.patterns = []
		rule_file = "data/clue_patterns.txt"
		if exists(rule_file):
			with open(rule_file, encoding="utf-8") as fin:
				for line in fin:
					p1, p2 = line.strip("\r\n").split("\t")
					self.patterns.append((re.compile(p1), re.compile(p2)))

		self.params = kwargs
	def prepclue(self, clue): return clue.strip(string.punctuation).strip()
	def equals(self, c1, c2): 
		if c1 == c2: return True
		for p1, p2 in self.patterns:
			if p1.match(c1) and p2.match(c2): return True
			if p2.match(c1) and p1.match(c2): return True
		return False
	def convertdoc(self, doc):
		for k, v in special_tokens.items(): doc = doc.replace(k, v)
		return doc
	def recoverdoc(self, cdoc):
		for k, v in special_tokens.items(): cdoc = cdoc.replace(v, k)
		return cdoc
	def fit2grid(self, ss):
		return re.sub("[%s]+"%string.punctuation, "", ss).upper()
	def initialize(self):
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
	def construct(self, dirn):
		''' 
		Construct the ES database with seen clue files
		'''
		docs = {}
		if self.dbname.startswith("cluedb"):
			print("loading from", dirn)
			for fn in os.listdir(dirn):
				with open(join(dirn, fn), encoding="utf-8") as fin:
					for line in fin:
						doc, strings = line.rstrip("\r\n").split("\t", 1)
						docs.setdefault(doc.replace("\t"," "), []).extend(strings.split(" "))
		elif self.dbname == "dictionary":
			with open(dirn, encoding="utf-8") as fin:
				for line in fin:
					item = json.loads(line.strip())
					for g in item["gloss"]:
						if not g.startswith('"'):
							docs.setdefault(g, []).extend(item["words"])
					
		print(len(docs), "unique clues")
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
					"text": self.convertdoc(doc), 
					"words": " ".join(strings), 
					"lengths": sorted(list(lens))
				}}
				if cnt % 1000 == 0 or not docs: print("%7d\t%.2f"%(cnt, time.time()-t0))
			print(tot, "clue-answer pairs")
		helpers.bulk(self.es, (item for item in data_gen(docs)))
		for item in data_gen(docs): pass
	def generate(self, clue, length, limit=None, blacklist=set(), **kwargs):
		''' 
		Generating answers given clues, with ES retrieving and rule scoring/normalization
		'''
		if limit is None: limit = self.limit
		ctokens = tokenize(clue)
		blacklist = blacklist | {string2fill(t) for t in ctokens} # should not include any words in the clue
		cands = self.retrieve(clue, length, limit, blacklist, **kwargs)
		if not cands: return []

		maxS = max(s for _, s, _ in cands)
		minS = min(s for _, s, _ in cands) if len(cands)>=limit else 0
		W = Z = 1; B = 0 # weight, normalizer, and base score
		if self.norm: # normalized by the length of query
			if self.norm > 0: # norm by length
				Z = len(ctokens)
				W = self.norm
			else: # norm by idf
				Z = sum([self.idf.get(t, self.max_idf) for t in ctokens])
				if self.upper == 1: # range from 0 ~ (max-min)/max
					Z = max(Z, maxS)
				elif self.upper == 2: # range from 0 ~ 1
					Z = max(Z, maxS) - minS
					if Z == 0: Z + 1e-9
				W = -self.norm
			# if maxS > minS and self.upper:
			# 	W = min(self.upper/(maxS-minS), W)
		
		# cdepunc = self.prepclue(clue)
		# for i in range(cands):
		# 	cdp = cands[i][2]
		# 	if self.equals(cdepunc, cdp): 
		# 		cands[i][1] = max(Z, maxS)

		B = minS/Z if self.base<0 else self.base # base score
		cands = [(w, W*max(s/Z-B, 0), c) for w, s, c in cands]
		return cands
	def retrieve(self, clue, length, limit=50, blacklist=set(), pagesize=50, target_clue=None, freq_black={}):
		''' 
		Retrieving seen clues from ES
		'''
		ret = []
		exs = blacklist
		fb = freq_black.copy() # block some answers certain times
		pfrom = 0
		while len(ret) < limit or target_clue is not None:
			results = self.es.search(index=self.dbname, body={
				"query": {"bool": {
					"must": [
						{"match": {"text": self.convertdoc(clue)}}
					],
					"filter": [
						{"term": {"lengths": length}},
					]
					# "must_not": {"term": {"lengths": 10}}
				}},
				"from": pfrom,
				"size": pagesize,
			}, request_timeout=100)
			pfrom += pagesize
			hits = results["hits"]["hits"]
			for item in hits:
				ss = [string2fill(s)for s in item["_source"]["words"].split(" ")]
				ws = set()
				for s in ss:
					if len(s) == length and s not in exs:
						if fb.get(s, 0) > 0: fb[s] -= 1
						else: ws.add(s)
				if ws:
					item["_source"]["text"] = self.recoverdoc(item["_source"]["text"])
					for w in ws:
						ret.append((w, item["_score"], item["_source"]["text"]))
					exs |= ws
				if len(exs) >= limit and target_clue is None: break
				if item["_source"]["text"] == target_clue: return ret
			if len(hits) < pagesize or pfrom >= 10000: break
		return ret

class RetrAndMatch(ClueES):
	''' 
	Retrieving with BM25 and matching with neural networks
	'''
	@property
	def matcher(self):
		if not hasattr(self, "_matcher"):
			from .siamese import Siamese
			self._matcher = Siamese()
			self._matcher.load()
		return self._matcher
	def __init__(self, match_weight=25, **kwargs):
		super().__init__(**kwargs)
		self.match_weight = match_weight
	def generate(self, clue, length, limit=None, blacklist=set(), pagesize=None, norm=0, base=None):
		if limit is None: limit = self.limit
		ctokens = tokenize(clue)
		blacklist = blacklist | {string2fill(t) for t in ctokens} # should not include any words in the clue
		cands = self.retrieve(clue, length, limit, blacklist)
		if not cands: return cands
		samples = [(clue, c) for w, s, c in cands]
		scores = self.matcher.do_batch(samples)[0].flatten()
		inds = (-scores).argsort()
		cands = [(cands[i][0], scores[i]*abs(norm), cands[i][-1]) for i in inds]
		return cands
class RAM_CTR(RetrAndMatch):
	''' 
	Retrieving and matching with different matching models
	'''
	@property
	def matcher(self):
		if not hasattr(self, "_matcher"):
			from .siamese import Contrastive, ContrastiveMarginal, ContrastiveMixup, ContrastiveMixupM, CTRMMM
			if self.params.get("MGN"):
				self._matcher = ContrastiveMarginal(**self.params)
				self._matcher.load()
			elif self.params.get("MIX"):
				self._matcher = ContrastiveMixup(**self.params)
				self._matcher.load()
			elif self.params.get("MM"):
				self._matcher = ContrastiveMixupM(**self.params)
				self._matcher.load()
			elif self.params.get("MMM"):
				self._matcher = CTRMMM(**self.params)
				self._matcher.load()
			else:
				self._matcher = Contrastive(**self.params)
				if self.params.get("org"):
					print("loading original BERT")
					self.matcher.build_model()
					self.matcher.build_tokenizers()
				else: self._matcher.load()
		return self._matcher
	def generate(self, clue, length, limit=None, blacklist=set(), pagesize=None, **kwargs):
		if limit is None: limit = self.limit
		ctokens = tokenize(clue)
		blacklist = blacklist | {string2fill(t) for t in ctokens} # should not include any words in the clue
		cands = self.retrieve(clue, length, limit, blacklist)
		if not cands: return cands
		cdepunc = self.prepclue(clue)
		samples = []
		for w, s, c in cands:
			cdp = self.prepclue(c)
			if self.equals(cdepunc, cdp): c = clue
			samples.append((clue, c))
		scores = self.matcher.do_batch(samples)[0].flatten()
		inds = (-scores).argsort()
		cands = [(cands[i][0], scores[i], cands[i][-1]) for i in inds]

		maxS = max(s for _, s, _ in cands)
		minS = min(s for _, s, _ in cands) if len(cands)>=limit else 0
		W = Z = 1; B = 0 # weight, normalizer, and base score
		if self.norm: # normalized by the length of query
			if self.upper == 1: # range from 0 ~ (max-min)/max
				Z = max(Z, maxS)
			elif self.upper == 2: # range from 0 ~ 1
				Z = max(Z, maxS) - minS
				if Z == 0: Z + 1e-9
			W = abs(self.norm)

		B = minS/Z if self.base<0 else self.base # base score
		cands = [(w, W*max(s/Z-B, 0), c) for w, s, c in cands]
		return cands
class RAM_CTR_NN(RAM_CTR):
	''' 
	With score normalization tricks
	'''
	def generate(self, clue, length, limit=None, blacklist=set(), pagesize=None, **kwargs):
		if limit is None: limit = self.limit
		ctokens = tokenize(clue)
		blacklist = blacklist | {string2fill(t) for t in ctokens} # should not include any words in the clue
		cands = self.retrieve(clue, length, limit*2, blacklist)
		if not cands: return cands
		cdepunc = self.prepclue(clue)
		samples = []
		for w, s, c in cands:
			cdp = self.prepclue(c)
			if self.equals(cdepunc, cdp): c = clue
			samples.append((clue, c))
		scores = self.matcher.do_batch(samples)[0].flatten()
		inds = (-scores).argsort()
		cands = [(cands[i][0], scores[i], cands[i][-1]) for i in inds]

		W = abs(self.norm)
		base = cands[limit][1] if len(cands)>limit else 0
		Z = 1-base
		cands = [(w, (s-base)/Z*W, c) for w, s, c in cands][:limit]
		return cands
