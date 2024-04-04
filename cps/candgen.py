import sys, argparse, json, string, os, time, numpy as np, random, math
from collections import defaultdict
from os.path import join, exists
import threading
from queue import Queue

from .puz_utils import *
from .utils import BaseObject
from .cdbretr import ClueES, RetrAndMatch, RAM_CTR, RAM_CTR_NN
from .reward import Rewarder, ClfRewarder, RNRewarder, CTRewarder, MicroCTR, SMRewarder
#from .poscls import poscls
from .clueparser import poscls
from .wfeats import get_vocab, get_w2v, get_wprior
from .dycg import MultiWordGenerator, MWG


class CandidateGenerator(BaseObject):
	''' Sources
		1. ES from clue DB
		2. Wordnet gloss
		3. Wiki
		4. Vocabulary
		5.  Multiword Generator/Char Sequence Generator
	'''
	@property
	def params(self): return dict(super().params, **{
		"cluedb_limit": 50,
		"vocab_limit": 20,
		"cluedb_norm": 0.,
		"base_clue_score": -1.,
		"n_props": 1,
		"kb_limit": 0,
		"wn_limit": 0,
		"wn_thres": 0.,
		'bf_limit': 0,
		#"bm": 1, # minimal length of cand list to use base score
		"sv_size": 1000, # small vocabulary size for a certain length
		"max_word_len": 23,
		"min_phrase_char": 5, # min number of chars for phrase generation
		"pg_thres": 0.3, # max ratio of blank chars for phrase generation
		"clues_upper": 0,
		"clue_match": 0,
		"match_weight": 25,
	})
	@property
	def options(self): return dict(super().options, **{
		"idf_path": "data/idf.txt",
		"synset_path": "data/wordnet/synsets.txt",
		"clue_dir": "data/clues_full",
		"clue_dbname": "cluedb210415",
		"vocab_path": "data/dictionaries/vocab.txt",
		"unigram_path": "data/dictionaries/unigram_freq_filtered.csv",
		"stringfeat_path": "data/dictionaries/stringfeats.txt",
	})
	@property
	def features(self):  
		if not hasattr(self, "_features"): 
			self._features = {
				# reserved features
				"RSV0": 0, "RSV1": 1, "RSV2": 2, "RSV3": 3, "RSV4": 4, 
				"RSV5": 5, "RSV6": 6, 
				"KB": 7, # knolwedge base retrieval score
				"POStag2": 8, 
				"POStag1": 9,
				# clue based score [0. bm25], deprecated:[1. TF-IDF cos, 2. Edit Distance Similarity, 3. Exact Match, 4. punctuation-free exact match]
				"ClueMatch": 10,
				# word basic info:[0. whether is a complete word, 1. length of the word]
				"IsAWord": 11,
				"WordLen": 12,
				# word prior features: [0. #occurs from ClueDB, 1. #occurs from CrossWordGiant, 2. #occurs from unigram, 3. wiki title]
				"OccCDB": 13,
				"OccCWG": 14,
				"OccUNG": 15,
				"InWiki": 16,
				# clue-word score: [0. w2v similarity of word and clues, 1. bigram score for filling blanks (TODO), 2. pos tag (TODO)]
				"W2Vcos": 17,
				"BF": 18, # blank filling
				"WN": 19, # dictionary retrieval score
			}
		return self._features
	@property
	def initializations(self): return super().initializations + [
		self.init_model, # initialize the rewards model
		self.init_basics, # clue mapping, feature vector, etc.
		self.init_cluedb, # ES: clueDB
		self.init_vocab, # Vocabulary
		self.init_w2v, # word2vec and word priors
		self.init_poscls, # postag classifier
		self.init_wnretr, # Wordnet retriever
		self.init_kbretr, # ES: KB retriever
		self.init_bf, # init blank filler
		self.init_mwg, # init multi-word generator
	]
	def init_basics(self, args, **kwargs):
		self.dfv = np.zeros((len(self.features)), dtype="float32") # default (nonword) feature vector
		self.drw = self.predict(self.dfv) # default (nonword) reward
		self.drv = [0, 0, 0, 0] # default vector for retrieval scores: [clueDB, KB, BF, Dict, ]
	def init_cluedb(self, args, **kwargs):
		clueDBparams = {
			"dbname": self.clue_dbname, 
			"dirn": self.clue_dir, 
			"limit": self.cluedb_limit, 
			"norm": self.cluedb_norm, 
			"base": self.base_clue_score, 
			"upper": self.clues_upper,
			"idf_path": self.idf_path
		}
		if self.clue_match == 3: 
			self.clueES = RAM_CTR_NN(MM=1, n_negs=1, batch_size=340, **clueDBparams)
		elif self.clue_match == 2: 
			self.clueES = RAM_CTR(MM=1, n_negs=1, batch_size=340, **clueDBparams)
		elif self.clue_match == 1: 
			self.clueES = RetrAndMatch(match_weight=self.match_weight, **clueDBparams)
		else:
			self.clueES = ClueES(**clueDBparams)
	def init_vocab(self, args, sfile='intermediate/clues_index/fill_strings.txt', **kwargs):
		self._vocabulary = {string2fill(s) for line in open(sfile, encoding="utf-8") for s in line.strip("\r\n").split(" ")}
		print("Vocab Size", len(self._vocabulary)) # 27.2
		self.construct_vocab_bucket()
	def construct_vocab_bucket(self):
		''' 
		Contructing dictionary buckets for a certain length and one-letter constraint for quick word filtering during search
		'''
		self._fill_strings = list(self._vocabulary)
		self._fill_strings.sort()
		self._strings = []
		for i, s in enumerate(self._fill_strings):
			while len(s) >= len(self._strings):
				self._strings.append(defaultdict(set))
			for j, c in enumerate(s):
				self._strings[len(s)][(j, c)].add(i)
			self._strings[len(s)].setdefault(0, []).append(i)
		print("Vocab bucket constructed") # 50.8
	def init_w2v(self, args, **kwargs):
		self.wfreqs = get_wprior(pfeat_path=self.stringfeat_path)
		self.wd2id, self.wvecs  = get_w2v()

	def init_poscls(self, args, delay_poscls=False, **kwargs):
		if delay_poscls:
			print("delay the initialization of clue POS classifier")
		elif not hasattr(poscls, "_pmodel"): 
			poscls.load() # 70
			print("POS cls loaded.") # 52.3
	@property
	def w_mapping(self): return { # model weights from linear ranking model
		13: 1.2337192,
		14: 0.21331637,
		15: -0.17480089,
		16: 0.3103069,
		17: 8.274433,
	}
	def init_model(self, args, omega1=0.1, omega2=5, load_weights=0, **kwargs):
		if load_weights < 2:
			self.rewarder = Rewarder(**kwargs)
		elif load_weights < 3:
			self.rewarder = ClfRewarder(**kwargs)
		elif load_weights < 4:
			self.rewarder = RNRewarder(**kwargs)
		elif load_weights < 5:
			self.rewarder = CTRewarder(**kwargs) 
		elif load_weights < 6: 
			self.rewarder = MicroCTR(**kwargs) 
		else:
			self.rewarder = SMRewarder(**kwargs) 
		
		if load_weights:
			self.rewarder.load()
			try:
				self.weights, self.bias = self.rewarder._pmodel.get_weights()
				self.weights, self.bias = self.weights.flatten(), float(self.bias)
				print("Linear model, predict with CPU")
				self.predict = self.predict_cpu
			except Exception:
				print("Not linear model, predict with GPU")
				self.predict = self.predict_gpu
		else:
			self.bias = self.rewarder.bias
			self.weights = self.rewarder.weights.copy()
			self.weights[10] = 1
			self.weights[11] = omega2
			for k, v in self.w_mapping.items():
				self.weights[k] = v * omega1
			self.predict = self.predict_cpu
	def init_kbretr(self, args, **kwargs):
		if self.kb_limit:
			from .kbretr import KBR_DW
			self.kbretr = KBR_DW(
				n_props=self.n_props, 
				with_words=1, 
				vocab_path=self.vocab_path,
				idf_path = self.idf_path,
				synset_path = self.synset_path,
			)
	def init_wnretr(self, args, **kwargs):
		if self.wn_limit:
			from .dictretr_v2 import W2VNorm
			self.wnretr = W2VNorm(thres=self.wn_thres)
	def init_bf(self, args, **kwargs):
		if self.bf_limit:
			from .fillblank import BlankFiller
			self.bf = BlankFiller()
	def init_mwg(self, args, **kwargs):
		self.mwg = MWG(
			wfreq_path=self.unigram_path,
			vocabfile=self.vocab_path,
		)
		self.mw_buf = {}
	def generate_phrase(self, cseq):
		''' 
		Phrase generation
		'''
		if len(cseq) < self.min_phrase_char: return []
		nm = cseq.count("*")
		if nm==0 or nm / len(cseq) > self.pg_thres: return []
		ret = self.mw_buf.get(cseq)
		if ret is None:
			r, s = self.mwg.generate(cseq)
			r = "".join(r)
			ret = self.mw_buf[cseq] = (r, s)
		return [ret]
	def generate_async(self, pos, clue, length, limit, blacklist):
		ret = self.kbretr.generate(clue, length, limit, blacklist)
		self.queue.put_nowait((pos, ret))
	def check_async(self): return self.threads and len(self.threads) == self.queue.qsize()
	def update_cands(self): 
		words, vecs, positions = [], [], []
		while not self.queue.empty():
			(i, num), cands = self.queue.get_nowait()
			clue, length = self.clues[i][num]
			cpos = self.clue_pos[i, num]
			for word, kbscore, kbclue in cands:
				words.append(word)
				vecs.append(self.make_feature(word, clue, cpos, 0))
				positions.append((i, num))
		rews = self.predict(np.array(vecs)).tolist() if vecs else []
		for k in range(len(words)):
			i, num = positions[k]
			w = words[k]
			#self.cands_from_cluedb[i][num][w] = (0, None)
			self.rewards[i, num][w] = [rews[k], 0, 0]
			self.cluedb_cands_list[i][num].append((w, rews[k]))
			self.cands_bufs[i, num].clear()
		self.threads.clear()
	def update_rewards(self, score, grid, num2pos):
		for i in range(len(self.clues)):
			for num, (_, l) in self.clues[i].items():
				x, y = num2pos[num]
				word = get_string(grid, x, y, i, l)
				rew = self.rewards[i, num].get(word)
				if rew:
					rew[1] += score
					rew[2] += 1
	def prepare_candidates(self, pos, clue, length, use_kb, use_wn, use_bf, clue_bl={}):
		''' 
		Geneartion clue-dependent candidates before the search
		'''
		cands = self.clueES.generate(clue, length, freq_black=clue_bl.get(pos, {}))
		
		if use_bf:
			for w, s, e in self.bf.generate(clue, length, \
			limit=use_bf, blacklist={w for w, s, c in cands}):
				cands.append((w, 0, None))

		if use_kb and len(cands) < self.cluedb_limit + use_bf:
			if pos: # if the position of the clue is given, use async generation
				th = threading.Thread(target=self.generate_async, \
				args=(pos, clue, length, use_kb, {w for w, s, c in cands}))
				#th.start() 
				self.threads.append(th) # not start first
			else:
				for w, s, e in self.kbretr.generate(clue, length, \
				limit=use_kb, blacklist={w for w, s, c in cands}):
					cands.append((w, 0, None))
		if use_wn and len(cands) < self.cluedb_limit + use_bf:
			for w, s, g in self.wnretr.generate(clue, length, \
			limit=use_wn, blacklist={w for w, s, c in cands}):
				cands.append((w, 0, None))
		return cands
	def construct_clue_cand_list(self):
		self.cluedb_cands_list = []
		for i in range(2):
			candlist = []
			for num, cands in self.cands_from_cluedb[i].items():
				while num >= len(candlist): candlist.append(None)
				candlist[num] = [(w, self.rewards[i, num][w][0])for w in cands]
				candlist[num].sort(key=lambda x:-x[1])
			self.cluedb_cands_list.append(candlist)
	def postag_clues(self, clues):
		''' 
		Predicting the postag of the answers according to the clue with a LSTM classifier
		'''
		if not hasattr(poscls, "_pmodel"): 
			poscls.load()
			print("POS cls loaded.")
		pos_and_clues = [((i, num), clue) for i in range(len(clues)) \
			for num, (clue, length) in clues[i].items()]
		positions, clue_list = zip(*pos_and_clues)
		pos_distrib = poscls.do_distrib(clue_list)
		return dict(zip(positions, pos_distrib))
	def replace_ref(self, clue, selfp):
		for _ in range(5):
			ref = re_ref.search(clue.lower())
			if not ref: break
			s, t = ref.span()
			num, d = ref.groups()
			d, num = directions.index(d), int(num)
			#clue = "%s[%s]%s"%(clue[:s], self.clues[d][num] if (d, num) != selfp else "", clue[t:])
			clue = "%s[%s]%s"%(clue[:s], self.clues[d][num], clue[t:])
		return clue
	def initialize(self, clues, use_kb=None, use_wn=None, use_bf=None, async_start=False, clue_bl={}): # initialize when a game starts
		if use_kb is None: use_kb = self.kb_limit
		if use_wn is None: use_wn = self.wn_limit
		if use_bf is None: use_bf = self.bf_limit

		self.queue = Queue()
		self.threads = []
		if use_kb:
			self.kbretr.init_stats(None)

		# calculate postag of clues
		self.clue_pos = self.postag_clues(clues)
		
		# get cands from clue DB
		self.clues = clues
		self._rewards = {} # initalized rewards, for clear to reset the game without initializing again
		self.cands_from_cluedb = [{}, {}]
		for i in range(len(clues)):
			for num, (clue, length) in clues[i].items():
				# reference replacement
				clue = self.replace_ref(clue, (i, num))

				cands = self.prepare_candidates((i, num), clue, length, use_kb, use_wn, use_bf, clue_bl=clue_bl)

				cpos = self.clue_pos[i, num]
				words, vecs = [], []
				for word, score, sclue in cands:
					words.append(word)
					vecs.append(self.make_feature(word, clue, cpos, score))
				rews = self.predict(np.array(vecs)) if vecs else []
				self.cands_from_cluedb[i][num] = {
					w: (s, c) for w, s, c in cands
				}
				self._rewards[i, num] = {
					w: [r, 0, 0] # immediate reward, accumulated filled reward, times of filling
						for w, r in zip(words, rews)
				}
				# if i == 0 and num == 6 and "Tom Cruise" in clue:
				# 	print("Features\t%s"%"\t".join(f for f, _ in sorted(self.features.items(), key=lambda x:x[1])[10:]))
				# 	print("Weights\t%s"%"\t".join("%.4f"%w for w in self.weights[10:].tolist()))
				# 	for i in range(len(words)):
				# 		print("%s\t%s"%(words[i], "\t".join("%.4f"%v for v in vecs[i][10:].tolist())))
				# 	input()
		# other variables
		self.cands_bufs = {(i, num): {} for i in range(len(self.clues)) for num in self.clues[i]}
		self.cands_from_vocab = {}
		self.rewards = self.copy_rewards(self._rewards) #self._rewards.copy()

		for th in self.threads: th.start()
		self.construct_clue_cand_list()
		if self.threads and not async_start: # for scenarios with async jobs and not to start asynchronously
			print("Waiting for async jobs done")
			t0 = time.time()
			while not self.check_async(): pass 
			self.update_cands() # update candidates with async generation results
			# for repetition tests, update self._rewards to reset the game with self.clear()
			self._rewards.clear()
			self._rewards = self.copy_rewards(self.rewards) #self.rewards.copy()
			if use_kb:
				print("Time consuming of kb init (%.2f secs in total)"%(time.time()-t0))
				for key, (t, n) in self.kbretr.timings.items():
					print("%s:\t%.2f/%d=%s"%(key, t, n, t/n))
		elif self.threads: print("%d async jobs"%len(self.threads))
	def copy_rewards(self, rewards): # deep copy rewards
		return {pos:{w:r.copy() for w, r in wdict.items()} for pos, wdict in rewards.items()}
	def mask_candidates(self, answers, mask_ratio, masked_cands=[], seed=12138):
		# mask_ratio is used only when masked_cands is not given
		clue_indices = [(i, num) for i, clues in enumerate(self.clues) for num in clues]
		if not masked_cands:
			mask_num = int(len(clue_indices)*mask_ratio)
			random.seed(seed)
			masked_cands = random.sample(clue_indices, mask_num)
		ret = []
		for i, num in masked_cands:
			ans = answers[i][num]
			for j in range(len(self.cluedb_cands_list[i][num])):
				if ans == self.cluedb_cands_list[i][num][j][0]:
					self.cluedb_cands_list[i][num].pop(j)
					self._rewards[i, num].pop(ans)
					self.rewards[i, num].pop(ans)
					ret.append((i, num))
					break
		print("%d (%d) out of %d candidates eliminated"%(len(ret), len(masked_cands), len(clue_indices)))
		return ret
	def clear(self, verbose=False): # clear the states after a game is over
		self.rewards.clear()
		self.rewards = self.copy_rewards(self._rewards) # self._rewards.copy()
		n = 0
		for pos in self.cands_bufs: 
			n += len(self.cands_bufs[pos])
			self.cands_bufs[pos].clear()
		if verbose:
			print("%d cluedb cands buffer cleared"%n)
			print("Clearing %d dictionary generations"%(len(self.cands_from_vocab)))
			print("Clearing %d phrase generations"%(len(self.mw_buf)))
		self.cands_from_vocab.clear()
		self.mw_buf.clear()
		t0 = time.time()
		waits = 0
		while self.threads and self.queue.qsize() < len(self.threads):
			sys.stdout.write("\rClearing: waiting for async jobs done, %f secs"%(time.time()-t0))
			sys.stdout.flush()
			time.sleep(1)
			waits += 1
		if self.threads:
			self.threads.clear()
			self.queue.queue.clear()
		if waits: print()

	def pitch(self, pitch_words): # pitch candidate words into the tail
		for (i, num), words in pitch_words.items():
			cands = self.cluedb_cands_list[i][num]
			if cands:
				mr = cands[-1][1] # min reward
				for j in range(len(cands)):
					if cands[j][0] in words: cands[j] = (cands[j][0], mr-1e-6)
				cands.sort(key=lambda x:-x[1])
	def reward(self, samples):
		ret, feats, inds = 0, [], []
		for idx, (i, num, word) in enumerate(samples):
			r = self.rewards[i, num].get(word)
			clue = self.clues[i][num][0]
			cpos = self.clue_pos[i, num]
			if r is None:
				if "*" in word: ret += self.drw 
				else:
					feats.append(self.make_feature(word, clue, cpos)) 
					inds.append(idx) 
			else: ret += r[0]
		if feats:
			feats = np.array(feats)
			rews = self.predict(feats)
			ret += float(rews.sum())
			for k in range(len(inds)):
				i, num, word = samples[inds[k]]
				self.rewards[i, num][word] = [float(rews[k]), 0, 0]
		return ret
	def predict_cpu(self, features): # calculate the reward by features
		return features.dot(self.weights)+self.bias
	def predict_gpu(self, features):
		if len(features.shape) == 1: 
			return self.rewarder.predict(features[None,:]).flatten()[0]
		else:
			return self.rewarder.predict(features).flatten()
	def wordscore(self, word):
		return bool(word in self._vocabulary)
	def make_feature(self, word, clue, cluepos, score=0):
		lword = word.lower()
		#wposv = self.w_pos.get(lword, self.dpv)
		wposv = poscls.wpos_distrib(word)
		feat = self.dfv.copy()
		# word feature: 0~6
		#src = self._vocabulary.get(word, 0)
		# feat[:len(self.bitops)] = (src&self.bitops)/self.bitops
		# feat[len(self.bitops)] = self.wfreqs.get(word, 0)
		# clue based score: 10
		feat[10] = score
		# word basic info: 11~12
		feat[11] = int(word in self._vocabulary)
		feat[12] = len(word)
		# word prior feats
		feat[13:17] = self.wfreqs.get(word, [0, 0, 0, 0])
		# clue-word score: 17
		if lword in self.wd2id:
			vec = self.wd2id[lword]
			ts = [self.wd2id[t] for t in tokenize(clue) if t in self.wd2id]
			if ts:
				# max
				feat[17] = self.wvecs[ts].dot(self.wvecs[vec]).max()
				# avg
				# feat[17] = self.wvecs[ts].dot(self.wvecs[vec]).mean()
		#feat[19] = (wposv*cluepos).max()
		# pos tag score: 8~9
		for i in range(len(wposv)):
			feat[9-i] = (wposv[i]*cluepos[i]).max()
		return feat
	def make_features(self, samples):
		feats = []
		for i, num, word in samples:
			mscore = self.cands_from_cluedb[i][num].get(word, [self.drv, None])[0]
			clue, _ = self.clues[i][num]
			feats.append(self.make_feature(word, clue, self.clue_pos[i, num], mscore))
		ret = np.array(feats, dtype="float32")
		feats.clear()
		return ret	
	def get_from_vocabulary(self, num, i, pattern):
		''' 
		Vocabulary generation
		'''
		if (i, num, pattern) in self.cands_from_vocab: ret = self.cands_from_vocab[i, num, pattern]
		else:
			clue = self.clues[i][num][0]
			ret = []
			cand = []
			for t, c in enumerate(pattern):
				if c != "*":
					cs = self._strings[len(pattern)][(t, c)]
					cand.append(cs)
			if cand: 
				cand = set.intersection(*cand)
			if cand and len(cand) <= self.vocab_limit: 
				words, vecs = [], []
				for wid in cand:
					wd = self._fill_strings[wid]
					words.append(wd)
					vecs.append(self.make_feature(wd, clue, self.clue_pos[i, num]))
				rews = self.predict(np.array(vecs)).tolist()
				for j in range(len(rews)):
					w, r = words[j], rews[j]
					ret.append((w, r))
					self.rewards[i, num][w] = [r, 0, 0]
				ret.sort(key=lambda x:-x[1])
			self.cands_from_vocab[i, num, pattern] = ret
		return ret
	def generate(self, num, i, pattern, timings=None):
		# from generated candidates
		# if timings:
		# 	timings["\tclueG"][1] += 1
		# 	timings["\tclueG"][0] -= time.time()
		cbufs = self.cands_bufs[i, num]
		ret = cbufs.get(pattern, None)
		if ret is None:
			ret = [] # word, reward
			for word, rew in self.cluedb_cands_list[i][num]:
				if match(word, pattern):
					ret.append((word, rew))
			cbufs[pattern] = ret
		# if timings:
		# 	timings["\tclueG"][0] += time.time()
		
		# from dictionary
		if not ret:
			# if timings:
			# 	timings["\tvocabG"][1] += 1
			# 	timings["\tvocabG"][0] -= time.time()
			for word, rew in self.get_from_vocabulary(num, i, pattern):
				ret.append((word, rew))
			# if timings:
			# 	timings["\tvocabG"][0] += time.time()
		
		# from multi-word generator
		# if not ret:
		# 	ret = self.generate_phrase(pattern)
		return ret
class CG_IRL(CandidateGenerator): # special version for IRLr, keep more information
	def initialize(self, clues, use_kb=None, use_wn=None, use_bf=None, async_start=False, answers=None): # initialize when a game starts
		if use_kb is None: use_kb = self.kb_limit
		if use_wn is None: use_wn = self.wn_limit
		if use_bf is None: use_bf = self.bf_limit

		self.queue = Queue()
		self.threads = []
		if use_kb:
			self.kbretr.init_stats(None)

		# calculate postag of clues
		self.clue_pos = self.postag_clues(clues)
		
		# get cands from clue DB
		self.clues = clues
		self.word_feats = {}
		self._rewards = {} # initalized rewards, for clear to reset the game without initializing again
		self.cands_from_cluedb = [{}, {}]
		for i in range(len(clues)):
			for num, (clue, length) in clues[i].items():
				# reference replacement
				clue = self.replace_ref(clue, (i, num))

				cands = self.prepare_candidates((i, num), clue, length, use_kb, use_wn, use_bf)

				cpos = self.clue_pos[i, num]
				words, vecs = [], []
				for word, score, sclue in cands:
					words.append(word)
					vecs.append(self.make_feature(word, clue, cpos, score))
				if answers and answers[i][num] not in words: # IRL setting, ignore the influence of candidate generation
					words.append(answers[i][num])
					vecs.append(self.make_feature(answers[i][num], clue, cpos, 0))
				rews = self.predict(np.array(vecs)) if vecs else []
				self.word_feats[i, num] = {
					w: v for w, v in zip(words, vecs)
				}
				self.cands_from_cluedb[i][num] = {
					w: (s, c) for w, s, c in cands
				}
				self._rewards[i, num] = {
					w: [r, 0, 0] # immediate reward, accumulated filled reward, times of filling
						for w, r in zip(words, rews)
				}
				# if i == 0 and num == 6 and "Tom Cruise" in clue:
				# 	print("Features\t%s"%"\t".join(f for f, _ in sorted(self.features.items(), key=lambda x:x[1])[10:]))
				# 	print("Weights\t%s"%"\t".join("%.4f"%w for w in self.weights[10:].tolist()))
				# 	for i in range(len(words)):
				# 		print("%s\t%s"%(words[i], "\t".join("%.4f"%v for v in vecs[i][10:].tolist())))
				# 	input()
		# other variables
		self.cands_bufs = {(i, num): {} for i in range(len(self.clues)) for num in self.clues[i]}
		self.cands_from_vocab = {}
		self.rewards = self.copy_rewards(self._rewards) #self._rewards.copy()

		for th in self.threads: th.start()
		self.construct_clue_cand_list()
		if self.threads and not async_start: # for scenarios with async jobs and not to start asynchronously
			print("Waiting for async jobs done")
			t0 = time.time()
			while not self.check_async(): pass 
			self.update_cands() # update candidates with async generation results
			# for repetition tests, update self._rewards to reset the game with self.clear()
			self._rewards.clear()
			self._rewards = self.copy_rewards(self.rewards) #self.rewards.copy()
			if use_kb:
				print("Time consuming of kb init (%.2f secs in total)"%(time.time()-t0))
				for key, (t, n) in self.kbretr.timings.items():
					print("%s:\t%.2f/%d=%s"%(key, t, n, t/n))
		elif self.threads: print("%d async jobs"%len(self.threads))
	def reward(self, samples):
		ret, feats, inds = 0, [], []
		for idx, (i, num, word) in enumerate(samples):
			r = self.rewards[i, num].get(word)
			clue = self.clues[i][num][0]
			cpos = self.clue_pos[i, num]
			if r is None:
				if "*" in word: ret += self.drw 
				else:
					feat = self.make_feature(word, clue, cpos)
					self.word_feats[i, num][word] = feat
					feats.append(feat) 
					inds.append(idx) 
			else: ret += r[0]
		if feats:
			feats = np.array(feats)
			rews = self.predict(feats)
			ret += float(rews.sum())
			for k in range(len(inds)):
				i, num, word = samples[inds[k]]
				self.rewards[i, num][word] = [float(rews[k]), 0, 0]
		return ret
	def clear(self, verbose=False): # clear the states after a game is over
		self.word_feats.clear()
		self.rewards.clear()
		self.rewards = self.copy_rewards(self._rewards) #self._rewards.copy()
		n = 0
		for pos in self.cands_bufs: 
			n += len(self.cands_bufs[pos])
			self.cands_bufs[pos].clear()
		if verbose:
			print("%d cluedb cands buffer cleared"%n)
			print("Clearing %d dictionary generations"%(len(self.cands_from_vocab)))
			print("Clearing %d phrase generations"%(len(self.mw_buf)))
		self.cands_from_vocab.clear()
		self.mw_buf.clear()
		t0 = time.time()
		waits = 0
		while self.threads and self.queue.qsize() < len(self.threads):
			sys.stdout.write("\rClearing: waiting for async jobs done, %f secs"%(time.time()-t0))
			sys.stdout.flush()
			time.sleep(1)
			waits += 1
		if self.threads:
			self.threads.clear()
			self.queue.queue.clear()
		if waits: print()
class CGLM(CandidateGenerator): # large vocab + multi-word generator
	def morph_score(self, cseq):
		if "*" in cseq or len(cseq) < self.min_phrase_char: return 0
		ret = self.mw_buf.get(cseq)
		if ret is None : 
			_, s = self.mwg.generate(cseq)
			ret = self.mw_buf[cseq] = (None, s)
		return ret[1]
	def wordscore(self, word):
		return bool(word in self._vocabulary) or self.morph_score(word)
	def init_vocab(self, args, sfile='intermediate/clues_index/fill_strings.txt', **kwargs):
		print("Loading vocab from", self.vocab_path)
		self._vocabulary = get_vocab(self.vocab_path)
		print("Vocab Size", len(self._vocabulary)) # 27.2
		self.construct_vocab_bucket()
	def make_feature(self, word, clue, cluepos, score=0):
		lword = word.lower()
		#wposv = self.w_pos.get(lword, self.dpv)
		wposv = poscls.wpos_distrib(word)
		feat = self.dfv.copy()
		# word feature: 0~6
		#src = word in self._vocabulary#self._vocabulary.get(word, 0)
		# feat[:len(self.bitops)] = (src&self.bitops)/self.bitops
		# feat[len(self.bitops)] = self.wfreqs.get(word, 0)
		# clue based score: 10
		feat[10] = score
		# word basic info: 11~12
		feat[11] = self.wordscore(word)#bool(src) or self.morph_score(word)
		feat[12] = len(word)
		# word prior feats
		feat[13:17] = self.wfreqs.get(word, [0, 0, 0, 0])
		# clue-word score: 17
		if lword in self.wd2id:
			vec = self.wd2id[lword]
			ts = [self.wd2id[t] for t in tokenize(clue) if t in self.wd2id]
			if ts:
				# max
				feat[17] = self.wvecs[ts].dot(self.wvecs[vec]).max()
				# avg
				# feat[17] = self.wvecs[ts].dot(self.wvecs[vec]).mean()
		#feat[19] = (wposv*cluepos).max()
		# pos tag score: 8~9
		for i in range(len(wposv)):
			feat[9-i] = (wposv[i]*cluepos[i]).max()
		return feat
	def generate(self, num, i, pattern, timings=None):
		ret = super().generate(num, i, pattern, timings)
		# from multi-word generator
		if not ret:
			ret = self.generate_phrase(pattern)
		return ret
class CGMF(CGLM): # multi-source features
	def update_cands(self): 
		words, vecs, positions = [], [], []
		while not self.queue.empty():
			(i, num), cands = self.queue.get_nowait()
			clue, length = self.clues[i][num]
			cpos = self.clue_pos[i, num]
			for word, kbscore, kbclue in cands:
				if word in self.cands_from_cluedb[i][num]:
					v = self.cands_from_cluedb[i][num][word][0]
				else: 
					v = self.drv.copy()
					self.cands_from_cluedb[i][num][word] = [v, "KB"]
				v[3] = kbscore
				words.append(word)
				vecs.append(self.make_feature(word, clue, cpos, v))
				positions.append((i, num))
		rews = self.predict(np.array(vecs)) if vecs else []
		for k in range(len(words)):
			i, num = positions[k]
			w = words[k]
			#self.cands_from_cluedb[i][num][w] = (0, None)
			if w in self.rewards[i, num]:
				self.rewards[i, num][w][0] = rews[k]
			else: self.rewards[i, num][w] = [rews[k], 0, 0]
			# if w in self._rewards[i, num]:
			# 	self._rewards[i, num][w][0] = rews[k]
			# else: self._rewards[i, num][w] = [rews[k], 0, 0]
			self.cands_bufs[i, num].clear()
		self.threads.clear()
		self.construct_clue_cand_list()
	def prepare_candidates(self, pos, clue, length, use_kb, use_wn, use_bf, clue_bl={}):
		cvecs = {}
		for w, s, c in self.clueES.generate(clue, length, freq_black=clue_bl.get(pos, {})):
			v = self.drv.copy()
			v[0] = s
			cvecs[w] = [v, c]
		
		if use_bf:
			for w, s, e in self.bf.generate(clue, length, limit=use_bf):
				if w in cvecs: cvecs[w][0][1] = s
				else:
					v = self.drv.copy()
					v[1] = s
					cvecs[w] = [v, "BF"]

		if use_wn and len(cvecs) < self.cluedb_limit + use_bf:
			for w, s, g in self.wnretr.generate(clue, length, limit=use_wn):
				if math.isnan(s): s = 0
				if w in cvecs: cvecs[w][0][2] = s
				else:
					v = self.drv.copy()
					v[2] = s
					cvecs[w] = [v, "DT"]

		if use_kb:
			if pos:
				th = threading.Thread(target=self.generate_async, \
					args=(pos, clue, length, use_kb, set()))
				#th.start() 
				self.threads.append(th) # not start first
			else:
				for w, s, e in self.kbretr.generate(clue, length, limit=use_kb):
					if w in cvecs: cvecs[w][0][3] = s
					else:
						v = self.drv.copy()
						v[3] = s
						cvecs[w] = [v, "KB"]

		return [(w, v, c)for w, (v, c) in cvecs.items()]
	def make_feature(self, word, clue, cluepos, score=None):
		if score is None: score = self.drv
		lword = word.lower()
		#wposv = self.w_pos.get(lword, self.dpv)
		wposv = poscls.wpos_distrib(word)
		feat = self.dfv.copy()
		# word feature: 0~6
		#src = word in self._vocabulary#self._vocabulary.get(word, 0)
		# feat[:len(self.bitops)] = (src&self.bitops)/self.bitops
		# feat[len(self.bitops)] = self.wfreqs.get(word, 0)
		# clue based score: 10
		feat[10] = score[0] # cluedb score
		feat[18] = score[1] # blank filler
		feat[19] = score[2] # dictionary retrieval
		feat[7] = score[3] # knowledge base retrieval
		# word basic info: 11~12
		feat[11] = self.wordscore(word)#bool(src) or self.morph_score(word)
		feat[12] = len(word)
		# word prior feats
		feat[13:17] = self.wfreqs.get(word, [0, 0, 0, 0])
		# clue-word score: 17
		if lword in self.wd2id:
			vec = self.wd2id[lword]
			ts = [self.wd2id[t] for t in tokenize(clue) if t in self.wd2id]
			if ts:
				# max
				feat[17] = self.wvecs[ts].dot(self.wvecs[vec]).max()
				# avg
				# feat[17] = self.wvecs[ts].dot(self.wvecs[vec]).mean()
		# pos tag score: 8~9
		for i in range(len(wposv)):
			feat[9-i] = (wposv[i]*cluepos[i]).max()
		return feat
