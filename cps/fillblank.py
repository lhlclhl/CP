
import numpy as np, os, time, json
from os.path import join, exists
from collections import defaultdict
from elasticsearch import Elasticsearch, helpers
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import to_array
from .utils import BaseObject
from .puz_utils import tokenize, string2fill, blank as blank_str

class BlankFiller(BaseObject):
	@property
	def options(self): return dict(super().options, **{
		"dbname": "wiki_titles",
		"src_file_path": "data/wiki/wikititles.txt",
		"new": False,
		"doc_limit": 20,
		"idf_path": "data/idf_wikititle.txt",
	})
	@property
	def bert_path(self): return "../bert/wwm_uncased_L-24_H-1024_A-16"
	@property
	def blank(self): return blank_str
	@property
	def initializations(self): 
		return super().initializations + [
			self.init_bert,
			self.init_es,
		]
	def init_es(self, args, **kwargs):
		self.es = Elasticsearch(**json.load(open("configs/elasticsearch.json")))
		if not self.es.indices.exists(self.dbname) or self.new:
			print("index doesn't exists, creating one")
			mapping = {
				'properties': {
					'title': {
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

			def data_gen(dirn):
				cnt = 0
				t0 = time.time()
				with open(dirn, encoding="utf-8") as fin:
					fin.readline()
					for line in fin:
						_id, remains = line.strip("\r\n").split("\t", 1)
						for item in remains.split("\t"):
							wikipedia_id, remain = item.split(",", 1)
							title, source = remain.rsplit(",", 1)
							yield {"_index": self.dbname, "_id": _id, "_source": {
								"redirect": int(source), 
								"title": title.replace("_", " "), 
							}}
						cnt += 1
						if cnt % 10000 == 0: print("%7d\t%.2f"%(cnt, time.time()-t0))
				print("%7d\t%.2f"%(cnt, time.time()-t0))
			helpers.bulk(self.es, (item for item in data_gen(self.src_file_path)), request_timeout=100)
	def init_bert(self, args, **kwargs):
		config_path = join(self.bert_path, 'bert_config.json')
		checkpoint_path = join(self.bert_path, 'bert_model.ckpt')
		dict_path = join(self.bert_path, 'vocab.txt')
		
		self.model = build_transformer_model(
			config_path=config_path, checkpoint_path=checkpoint_path, with_mlm=True
		)  # 建立模型，加载权重
		self.tokenizer = Tokenizer(dict_path, do_lower_case="uncased" in self.bert_path)  # 建立分词器

		token_ids, _ = self.tokenizer.encode(self.blank)
		self.blank_tokens = token_ids[1:-1]
	def find_blank(self, token_ids): # TODO: KMP (unnecessary)
		i = 0
		N = len(self.blank_tokens)
		while i < len(token_ids)-N+1:
			match = True
			for j in range(N):
				if token_ids[i+j] != self.blank_tokens[j]: 
					match = False
					break
			if match: return (i, i+N)
			i += 1
		return -1, -1
	def fill_by_bert(self, clue, length, limit=10, max_masked_words=1, blacklist={"AND", "FOR", "THE"}):
		ret = getattr(self, "bert_buf", {}).get(clue)
		if ret is not None: return ret

		if self.blank in clue:
			token_ids, segment_ids = self.tokenizer.encode(clue)
		else:
			token_ids, segment_ids = self.tokenizer.encode(clue, self.blank)
		
		tokens, segments, mlens = [], [], []
		s, t = self.find_blank(token_ids)
		if t < 0: return []
		for i in range(1, max_masked_words+1):
			tokens.append(token_ids[:s] + [self.tokenizer._token_dict["[MASK]"]] * i \
				+ token_ids[t:] + [self.tokenizer._token_dict["[PAD]"]] * (max_masked_words-i))
			segments.append(segment_ids[:s] + segment_ids[s:s+1] * i + segment_ids[t:]
				+ segment_ids[-1:] * (max_masked_words-i))
			mlens.append(i)
		# for token_ids in tokens:
		# 	print(self.tokenizer.ids_to_tokens(token_ids))	
		probas = self.model.predict([tokens, segments])
		preds = (-probas).argsort(axis=-1)
		
		words = {}
		for i in range(max_masked_words):
			for w in preds[i][s+1]:
				wd = self.tokenizer.id_to_token(w)
				fillstring = string2fill(wd)
				if len(fillstring) == length and fillstring not in blacklist:
					words[fillstring] = words.get(fillstring, 0) + float(probas[i][s+1][w])
					if len(words) >= limit: break
			if len(words) >= limit: break
		ret = [(w, s, "BERT") for w, s in words.items()]
		ret.sort(key=lambda x:-x[1])
		if hasattr(self, "bert_buf"): self.bert_buf[clue] = ret
		return ret
	def retrieve(self, query, ifrom=0, limit=None):
		if limit is None: limit = self.doc_limit
		key = "%s: %d"%(query, ifrom)
		ret = getattr(self, "wiki_buf", {}).get(key)
		if ret is not None: return ret

		results = self.es.search(index=self.dbname, body={
			"query": {"bool": {
				"must": [
					{"match": {"title": query}}
				],
				"filter": [
				]
				# "must_not": {"term": {"lengths": 10}}
			}},
			"from": ifrom,
			"size": limit,
		}, request_timeout=100)
		ret = results["hits"]["hits"]
		ret.sort(key=lambda x:(-x["_score"], x["_source"]["redirect"]))
		if hasattr(self, "wiki_buf"): self.wiki_buf[key] = ret
		return ret
	def fill_by_wiki(self, clue, length, limit=10, blacklist=set()):
		if self.blank not in clue: return []
		
		tokens = tokenize(clue)
		exs = blacklist | {string2fill(t) for t in tokens}
		if self.blank not in tokens: return []
		bid = tokens.index(self.blank)
		pre = tokens[bid-1] if bid > 0 else None
		post = tokens[bid+1] if bid+1 < len(tokens) else None
		
		ret = []
		cnt = 0
		while len(ret) < limit and cnt < limit*10:
			items = self.retrieve(clue.replace(self.blank, ""), ifrom=cnt)
			for item in items:
				tts = tokenize(item["_source"]["title"])
				cands = []
				for i in range(len(tts)):
					fs = string2fill(tts[i])
					if len(fs) == length and fs not in exs:
						# if (i == 0 or pre == tts[i-1]) and (i+1 == len(tts) or post == tts[i+1]):
						# 	ret.append((fs, item["_score"], item["_source"]["title"]))
						score = item["_score"]
						if i > 0 and pre == tts[i-1]: score *= 2
						if i+1 < len(tts) and post == tts[i+1]: score *= 2
						cands.append((fs, score))
				if cands:
					cands.sort(key=lambda x:-x[1])
					if cands[0][1] > cands[-1][1]:
						cands = cands[:1]
					for f, s in cands:
						ret.append((f, s, item["_source"]["title"]))
						exs.add(f)
			cnt += len(items)
			if not items: break
		return ret[:limit]
	def generate(self, clue, length, limit=10, blacklist=set()):
		ret = self.fill_by_wiki(clue, length, limit=limit, blacklist=blacklist)
		# if len(ret) < limit:
		# 	ret += self.fill_by_bert(clue, length, limit-len(ret))
		return ret
class BlankFillerProb(BlankFiller):
	@property
	def params(self): return dict(super().params, **{
		"smooth": 0.1,
		"eta": 0.
	})
	@property
	def initializations(self): 
		return super().initializations + [
			self.init_idf,
			self.init_buf,
		]
	def init_idf(self, args, **kwargs):
		self.idf = {}
		with open(self.idf_path, encoding="utf-8") as fin:
			for line in fin:
				word, f = line.strip().split("\t")
				self.idf[word] = float(f)
	def init_buf(self, args, buf_dir="outputs/fillblanks", **kwargs): 
		bert_buf_file = join(buf_dir, "bert_buf.json")
		if exists(bert_buf_file):
			with open(bert_buf_file, encoding="utf-8") as fin:
				self.bert_buf = json.load(fin)
		else: self.bert_buf = {}
		wiki_buf_file = join(buf_dir, "wiki_buf.json")
		if exists(wiki_buf_file):
			with open(wiki_buf_file, encoding="utf-8") as fin:
				self.wiki_buf = json.load(fin)
		else: self.wiki_buf = {}
	def save_buf(self, buf_dir="outputs/fillblanks"):
		with open(join(buf_dir, "bert_buf.json"), "w", encoding="utf-8") as fout:
			json.dump(self.bert_buf, fout)
		with open(join(buf_dir, "wiki_buf.json"), "w", encoding="utf-8") as fout:
			json.dump(self.wiki_buf, fout)
	def fill_by_wiki(self, clue, length, limit=10, blacklist=set()):
		tokens = tokenize(clue)
		exs = blacklist | {string2fill(t) for t in tokens}
		try: bid = tokens.index(self.blank)
		except Exception:
			print(clue, tokens)
			return []
		pre = tokens[bid-1] if bid > 0 else None
		post = tokens[bid+1] if bid+1 < len(tokens) else None
		denom = int(pre is not None) + int(post is not None)
		maxs = sum([self.idf.get(t, self.idf["MAX"]) for t in tokens if t != self.blank])
		
		ret = []
		cnt = 0
		while len(ret) < limit and cnt < limit*10:
			items = self.retrieve(clue.replace(self.blank, ""), ifrom=cnt)
			for item in items:
				tts = tokenize(item["_source"]["title"])
				cands = []
				for i in range(len(tts)):
					fs = string2fill(tts[i])
					if len(fs) == length and fs not in exs:
						# if (i == 0 or pre == tts[i-1]) and (i+1 == len(tts) or post == tts[i+1]):
						# 	ret.append((fs, item["_score"], item["_source"]["title"]))
						score = item["_score"]
						if score > maxs: maxs = score
						prob = score / maxs * (
							(
								int(i > 0 and pre == tts[i-1]) + 
								int(i+1 < len(tts) and post == tts[i+1])
							) / denom * (1-self.smooth) + self.smooth
						)
						cands.append((fs, prob))
				#print(item["_source"], tts, cands)
				if cands:
					cands.sort(key=lambda x:-x[1])
					if cands[0][1] > cands[-1][1]:
						cands = cands[:1]
					for f, s in cands:
						ret.append((f, s, item["_source"]["title"]))
						exs.add(f)
			cnt += len(items)
			if not items: break
		ret.sort(key=lambda x:-x[1])
		return ret[:limit]
	def generate(self, clue, length, limit=10, blacklist=set(), eta=None):
		if self.blank not in clue: return []
		if eta is None: eta = self.eta
		cands = defaultdict(float)
		wiki_res = self.fill_by_wiki(clue, length, limit=limit, blacklist=blacklist)
		ss_wiki = sum([s for w, s, c in wiki_res])
		for w, s, c in wiki_res:
			cands[w] = s/ss_wiki * (1-eta)
		if eta:
			bert_res = self.fill_by_bert(clue, length, limit)
			ss_bert = sum([s for w, s, c in bert_res])
			for w, s, c in bert_res:
				cands[w] += s/ss_bert * eta
		
		return [(w, s, "FB") for w, s in sorted(cands.items(), key=lambda x:-x[1])[:limit]]

def test_fill(odir="outputs/fillblanks", limit=100, cluefile="data/clues_test.txt"):
	if not exists(odir): os.makedirs(odir)
	bf = BlankFillerProb()
	t = n = 0
	with open(cluefile, encoding="utf-8") as fin, \
	open(join(odir, "wiki_fill_res.txt"), "w", encoding="utf-8") as fout:
		for line in fin:
			date, clue, ans = line.strip("\r\n").split("\t")
			index, clue = clue.split(".", 1)
			if "___" in clue:
				n += 1
				ret = bf.generate(clue, len(ans))
				rank = -1
				for i, (w, s, _) in enumerate(ret):
					if ans == w:
						rank = i+1
						break
				t += rank > 0
				fout.write("%s\t%s. %s\t%s\t%+d\n"%(date, index, clue, ans, rank))
				for w, s, _ in ret:
					fout.write("\t%s\t%.4e\n"%(w, s))
	print("accuracy=%d/%d=%.4f"%(t, n, t/n))
def get_rank(res, ans):
	rank = 999999
	for i in range(len(res)):
		k, v, _ = res[i]
		if k.lower() == ans.lower(): 
			rank = i
			break
	if rank >= len(res): return rank+1

	ans, score, _ = res[rank]
	rank = 0
	for k, v, _ in res:
		if v >= score: rank += 1
		else: break
	return rank
def debug_fill(odir="outputs/fillblanks", limit=100, cluefile="data/clues_for_test.txt"):
	if not exists(odir): os.makedirs(odir)
	bf = BlankFillerProb(smooth=0.01)
	t = n = n1 = n2 = 0
	rr1 = rr2 = 0.
	m1name, m2name = "WK", "SM"

	# def load_buf(fn):
	# 	BFbuf = {}
	# 	with open(fn, encoding="utf-8") as fin:
	# 		for line in fin:
	# 			item = json.loads(line)
	# 			BFbuf[item["clue"]] = item["res"]
	# 	return BFbuf
	# WKbuf = load_buf("outputs/candgen_mix/BF_WK.json")
	# SMbuf = load_buf("outputs/candgen_mix/BF_SM.json")

	with open(cluefile, encoding="utf-8") as fin, \
	open(join(odir, "debug_res_WK-SM.txt"), "w", encoding="utf-8") as fout:
		for line in fin:
			item = json.loads(line.strip())
			clue, ans = item["clue"], item["answer"]

			if "___" in clue:
				n += 1
				m1_res = bf.generate(clue, len(ans), limit=limit, eta=0)#WKbuf.get(clue, [])#bf.fill_by_wiki(clue, len(ans), limit=limit)
				m2_res = bf.generate(clue, len(ans), limit=limit, eta=0.04)#SMbuf.get(clue, [])#bf.fill_by_bert(clue, len(ans), limit=limit)#super(BlankFillerProb, bf).fill_by_wiki(clue, len(ans), limit=limit)
				m1_rank = get_rank(m1_res, ans)
				m2_rank = get_rank(m2_res, ans)
				rr1 += 1/m1_rank
				rr2 += 1/m2_rank
				if m1_rank != m2_rank:
					if m1_rank < m2_rank:
						fout.write("%s better\n"%m1name)
						n1 += 1
					else: 
						fout.write("%s better\n"%m2name)
						n2 += 1
					fout.write("%s\t%s\t%+d\t%+d\n"%(clue, ans, m1_rank, m2_rank))	
					for i in range(max(len(m1_res), len(m2_res))):
						r1, s1 = (m1_res[i][0], m1_res[i][1]) if i < len(m1_res) else ("", 0)
						r2, s2 = (m2_res[i][0], m2_res[i][1]) if i < len(m2_res) else ("", 0)
						fout.write("%s: %.4e\t%s: %.4e\n"%(r1, s1,\
							r2, s2))
						if r2 == ans or r1 == ans: break
					fout.write("\n")
				# print("prob",  wiki_res_prob)
				# print("score", wiki_res)
				# input()
				# rank = -1
				# for i, (w, s, _) in enumerate(ret):
				# 	if ans == w:
				# 		rank = i+1
				# 		break
				# t += rank > 0
				# fout.write("%s\t%s. %s\t%s\t%+d\n"%(date, index, clue, ans, rank))
				# for w, s, _ in ret:
				# 	fout.write("\t%s\t%.4e\n"%(w, s))
	bf.save_buf()
	print("total %d samples, %d %s better, %d %s better"%(n, n1, m1name, n2, m2name))
	print("%s MRR=%.4f\t%s MRR=%.4f"%(m1name, rr1/n*100, m2name, rr2/n*100))
def tune(odir="outputs/fillblanks", limit=100, cluefile="data/clues_for_test.txt", \
etas=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4], smooths=[1, 2, 3, 4]):
	bf = BlankFillerProb()
	n = tot = 0
	hits = {sm: {int(r*100):0 for r in etas} for sm in smooths}
	with open(cluefile, encoding="utf-8") as fin, \
	open(join(odir, "tuning_norm.txt"), "w", encoding="utf-8") as fout:
		for line in fin:
			item = json.loads(line.strip())
			clue, ans = item["clue"], item["answer"]
			if "___" in clue:
				n += 1
				for sm in hits:
					bf.smooth = 10**(-sm)
					for r in hits[sm]:
						ret = bf.generate(clue, len(ans), limit, eta=r/100)
						rank = get_rank(ret, ans)
						if rank <= len(ret):
							hits[sm][r] += 1/rank
			tot += 1
		restr = "samples\t%d/%d=%.2f%%\n"%(n, tot, n/tot*100)
		fout.write(restr)
		print(restr, end="")
		for s, h in sorted(hits.items()):
			for r, t in h.items():
				restr = "smooth=%d\teta=%.2f\tMRR=%f%%\n"%(s, r/100, t*100/n)
				fout.write(restr)
				print(restr, end="")
			fout.write("\n")
			print()
if __name__ == "__main__":
	
	# clue, length = "\"So ___?\"", 4 # ("___ fraiche", 5, max_masked_words=1) # ("\"___: Uprising\" (Disney animated series)", 4)
	# ret = bf.fill(clue, length, max_masked_words=1)
	# for r in ret: print(r)

	# bf = BlankFiller()
	# ret = bf.fill_by_wiki("\"___: Uprising\" (Disney animated series)", 4)
	# for r in ret: print(r)

	#test_fill()
	debug_fill()
	#tune(etas=[i/100 for i in range(20)])
	#debug_fill()
	# bf = BlankFillerProb()
	# clue = "\"Don't ___!\""
	# for item in bf.retrieve(clue, 20):
	# 	print(item["_score"], item["_source"])
	# for item in bf.fill_by_wiki(clue, 3):
	# 	print(item)
	# for (clue, length) in [("___ hot", 4), ("__ spit", 4)]:
	# 	print(clue, length)
	# 	for r in bf.generate(clue, length):
	# 		print("\t", r)
