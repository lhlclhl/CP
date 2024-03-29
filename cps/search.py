import random, math, os, time, json, traceback, sys, argparse, inspect
import numpy as np, unidecode, h5py
from collections import defaultdict, OrderedDict
from itertools import count
from os.path import exists, join, split
#from functools import lru_cache 

from .puz_utils import *
from .utils import *
from .candgen import *

class CPSolver(BaseObject):
	@property
	def initializations(self): return super().initializations + [self.init_candgen]
	def init_candgen(self, args, clueDB=None, cache=1, generator="bm25", **kwargs):
		self.candgen = CandidateGenerator()
	def load_data(self, args, **kwargs): pass
	def initialize(self, pzlid, grid, clues, num2pos, candlimit=50, mask_ratio=0, answers=None, **kwargs):
		self.clues = clues
		self.num2pos = num2pos
		self.n_states = 0
		self.answers = answers
		self.candgen.initialize(clues, **kwargs)
		if self.mask_ratio > 0 or masked_cands: 
			self.masked_cands = self.candgen.mask_candidates(self.answers, self.mask_ratio, masked_cands)
		else: self.masked_cands = []
		self.cp_size = len(self.clues[0])+len(self.clues[1])
	def clear(self): pass
	def search(self, grid, time_limit=0, **kwargs):
		self.time_limit = time.time()+time_limit if time_limit else 1e11
		ms, bg, bf = self._search(0, grid, 0, grid)
		return self.postprocess(ms, bg)
	def postprocess(self, max_score, best_grid): return max_score, best_grid
	# @lru_cache(maxsize = 500000)
	def generate_candidates(self, unfilled):
		actions = []
		for i, num, string in unfilled:
			valid_words = []
			for word, score in self.candidates[i][num].items():
				if match(word, string):
					valid_words.append((word, score))
			#if not valid_words: return max_score, best_grid
			for w, s in valid_words:
				actions.append((len(valid_words), -s, num, i, w)) # 优先数量少的，然后是分数高的
		return actions
	def est_grid(self, grid):
		est = score = 0
		unfilled = []
		for i in range(len(self.candidates)):
			for num, cands in self.candidates[i].items():
				_, l = self.clues[i][num]
				word = get_string(grid, *self.num2pos[num], i, l)
				if "*" in word:
					est += max(cands.values()) if cands else 0
					unfilled.append((i, num, word))
				else:
					score += cands.get(word, 0)
		return score, est, unfilled
	def _search(self, levels, grid,max_score, best_grid):
		self.n_states += 1
		ss = concate_bytes(grid)
		best_fill = (levels, -ss.count("*"))
		if time.time() >= self.time_limit: return max_score, best_grid, best_fill
		score, est, unfilled = self.est_grid(grid)
		if score + est <= max_score: return max_score, best_grid, best_fill
		elif score > max_score:
			max_score = score
			best_grid = grid.copy()
		actions = self.generate_candidates(unfilled)
		actions.sort()
		for _, _, num, i, word in actions:
			ngrid = fill(grid, i, *self.num2pos[num], word)
			ms, bg, bf = self._search(levels+1, ngrid, max_score, best_grid)
			if ms > max_score:
				max_score, best_grid = ms, bg
			if bf > best_fill: best_fill = bf
		return max_score, best_grid, best_fill

class MCTS(CPSolver): 
	# hyper-parameters
	@property
	def tau(self): return 0.20 # temperature of default policy
	@property
	def w_lambda(self): return .08 # exploration factor
	@property
	def CGParams(self): return { # parameters for candidate generator
		"omega1": 0.7,

		"cluedb_limit": 50,
		"base_clue_score": -1,

		"vocab_limit": 200,
		"clues_upper": 2,
		"omega2": 15,

		"cluedb_norm": -26.39,
	}
	@property
	def foldscore(self): return 0

	# settings
	@property
	def mask_ratio(self): return 0 # ratio of candidates to mask
	@property
	def doPost(self): return 1 # whether to postprocess
	@property
	def CGType(self): return CGMF
	@property
	def candidates(self): return self.candgen.cands_from_cluedb

	# initializations
	def norm_weights(self):
		self.candgen.weights /= (self.candgen.weights**2).sum()**.5
	@property
	def reward_weights(self):
		return None
	@property
	def initializations(self): return super().initializations + [
		self.init_lambda
	]
	def init_lambda(self, args, n_samples=10, **kwargs):
		self.timings = {"est": [0, 0], "act": [0, 0]}
		rewards = []
		with open("data/puzzles/nyt.shuffle.txt", encoding="utf-8") as fin:
			for line in fin:
				puzzle = json.loads(line)
				try:
					grid, num2pos, pos2num, clues, answers, agrid = construct_grid(puzzle)
				except Exception as e: 
					continue
				if agrid.dtype != np.dtype("S1"):
					continue

				self.initialize(puzzle['date'], agrid, clues, num2pos, answers=answers, use_kb=0, use_wn=0, use_bf=0)
				reward, _ = self.est_grid(agrid)
				rewards.append(reward)
				self.candgen.clear()
				if len(rewards) >= n_samples: break
		self._lambda = sum(rewards)/len(rewards)
		print("average rewards in training set:", self._lambda)
		self._lambda *= self.w_lambda # normalized exploration factor in training set, used when variance control mechanism is not applied
	def init_candgen(self, args, **kwargs):
		self.candgen = self.CGType(**dict(self.CGParams, **kwargs))
		if self.reward_weights is not None:
			self.candgen.weights = self.reward_weights
		self.norm_weights()
	def initialize(self, pzlid, grid, clues, num2pos, candlimit=50, answers=None, masked_cands=[], **kwargs):
		self.clues = clues
		self.num2pos = num2pos
		self.n_states = 0
		self.answers = answers
		self.candgen.initialize(clues, **kwargs)
		if self.mask_ratio > 0 or masked_cands: 
			self.masked_cands = self.candgen.mask_candidates(self.answers, self.mask_ratio, masked_cands)
		else: self.masked_cands = []
		self.cp_size = len(self.clues[0])+len(self.clues[1])

		self.mappings = [[[None, None] for j in range(grid.shape[1])] for i in range(grid.shape[0])]
		for num, (_, length) in self.clues[0].items():
			x, y = self.num2pos[num]
			for k in range(x, x+length):
				self.mappings[k][y][0] = (num, length)
		for num, (_, length) in self.clues[1].items():
			x, y = self.num2pos[num]
			for k in range(y, y+length):
				self.mappings[x][k][1] = (num, length)
	def clear(self): 
		print("%d states, %d unique states"%(self.n_states, len(self.action_buf)))
		self.action_buf.clear()
		print("%d valued states"%(len(self.state_buf)))
		self.state_buf.clear()
		self.candgen.clear(verbose=True)
		self.n_states = 0

	# utilities
	def UCB(self, total_s_child, N_child, sqrtN, prob, base):
		Q = total_s_child/N_child if N_child else 1e9
		U = base * sqrtN * prob/(N_child+1)
		return Q + U
	def base(self, total_s, N, current_s):
		return self.w_lambda * (total_s / N - current_s)
	def sample_action(self, actions, num=1):
		return np.random.choice(len(actions), num, p=np.array([a[0] for a in actions]))[0]
	def get_state_hash(self, grid, blacklist, fold=0):
		fold = str(fold) if fold else ""
		ss = concate_bytes(grid)
		black = "".join(word for _, _, word in sorted(blacklist))
		return "%s|%s|%s"%(fold, ss, black)
	def get_actions(self, grid, blacklist, fold, ss=None, mode=0): # mode0: selection; mode1: simulation
		score, unfilled = self.est_grid(grid)
		self.timings["act"][1] += 1
		self.timings["act"][0] -= time.time()
		actions = []
		for i, num, string in unfilled: 
			cands = self.candgen.generate(num, i, string, timings=self.timings)
			ff = fold
			while len(cands) > ff and (i, num, cands[ff][0]) in blacklist: ff += 1
			if ff < len(cands):
				w, s = cands[ff]
				second_largest_score = cands[ff+1][1] if len(cands) > ff+1 else 0
				g = None # lazy strategy: initialize ngrid to None #fill(grid, i, *self.num2pos[num], w)#
				actions.append([second_largest_score-s, num, i, w, g, blacklist, 0]) 
		if mode==0 and actions: 
			actions.append([self.foldscore, 0, 0, "", grid, blacklist, fold+1])
		self.timings["act"][0] += time.time()
		ranked_actions = self.rank_actions(grid, actions)
		return ranked_actions, score
	def rank_actions(self, grid, actions):
		# convert score of actions to probability
		probs = np.array([-a[0] for a in actions])
		probs = np.exp(probs/self.tau)
		probs /= probs.sum()
		for i in range(len(actions)):
			actions[i][0] = probs[i]
		actions.sort(reverse=True)
		return actions
	def est_grid(self, grid):
		self.timings["est"][1] += 1
		self.timings["est"][0] -= time.time()
		reward = 0
		unfilled = []
		samples = []
		for i in range(len(self.clues)):
			for num, (_, l) in self.clues[i].items():
				x, y = self.num2pos[num]
				word = get_string(grid, x, y, i, l)
				if "*" in word:
					unfilled.append((i, num, word))
					reward += self.candgen.drw # reward of nonwords
				else:
					samples.append((i, num, word))
		reward += self.candgen.reward(samples)
		self.timings["est"][0] += time.time()
		return reward / self.cp_size, unfilled

	# search processes
	def _simulate(self, depth, grid, blacklist=set(), fold=0, **kwargs):
		self.n_states += 1
		actions, score = self.get_actions(grid, blacklist, fold, mode=1)
		if time.time() >= self.time_limit: return score, grid, score
		max_score, best_grid = score, grid
		if actions:
			r = self.sample_action(actions)
			p, num, i, word, ngrid, nbl, fd = actions[r]
			if ngrid is None: 
				ngrid = actions[r][4] = fill(grid, i, *self.num2pos[num], word)
			self.action_list.append((num, i, word, fd, actions))
			ms, bg, _ = self._simulate(depth+1, ngrid, blacklist|set(nbl), fd)
			if ms > max_score:
				max_score, best_grid = ms, bg
		return max_score, best_grid, score
	def _select_and_expand(self, depth, grid, blacklist=set(), fold=0, **kwargs):
		self.n_states += 1
		ss = self.get_state_hash(grid, blacklist, fold)
		if ss in self.state_buf: # select
			state_item = self.state_buf[ss]
			total_s, N, score = state_item
			
			if time.time() >= self.time_limit: return score, grid
			
			# actions = self.get_actions(grid, None, blacklist, fold)
			actions, _ = self.get_actions(grid, blacklist, fold, ss)

			if not actions: return score, grid

			maxUCB, best_child = -1e9, None
			sqrtN = math.sqrt(N)
			base = self.base(total_s, N, score)
			for a, (p, num, i, word, ngrid, nbl, fd) in enumerate(actions):
				if ngrid is None: 
					ngrid = actions[a][4] = fill(grid, i, *self.num2pos[num], word)
				key = self.get_state_hash(ngrid, blacklist|nbl, fd)
				if key in self.state_buf: total_s_child, N_child, _ = self.state_buf[key]
				else: total_s_child, N_child, _ = 0, 0, 0
				UCB = self.UCB(total_s_child, N_child, sqrtN, p, base)
				if UCB > maxUCB:
					maxUCB = UCB
					best_child = a
			if best_child is not None:
				s, num, i, word, ngrid, nbl, fd = actions[best_child]
				#if self.fdeb and self.answers: self.log_fill(num, i, word)
				self.action_list.append((num, i, word, fd, actions))
				reward, best_grid = self._select_and_expand(depth+1, ngrid, blacklist|nbl, fd) # same level, next depth
				state_item[0] += reward
				state_item[1] += 1
			else: reward, best_grid = score, grid
		else: # expand
			reward, best_grid, score = self._simulate(depth, grid, blacklist, fold)
			self.state_buf[ss] = [reward, 1, score] # total reward, visit times, score
		return reward, best_grid
	def _lds(self, depth, grid, blacklist=set(), fold=0, discrepancy_limit=0, **kwargs):
		actions, score = self.get_actions(grid, blacklist, fold)
		if time.time() >= self.time_limit: return score, grid
		max_score, best_grid = score, grid
		if actions:
			r = 0
			p, num, i, word, ngrid, nbl, fd = actions[r]
			if ngrid is None: 
				ngrid = actions[r][4] = fill(grid, i, *self.num2pos[num], word)
			ms, bg = self._lds(depth+1, ngrid, blacklist|set(nbl), fd, discrepancy_limit)
			if ms > max_score:
				max_score, best_grid = ms, bg
			if len(blacklist) < discrepancy_limit and (i, num, word) not in blacklist:
				ms, bg = self._lds(depth, grid, blacklist|{(i, num, word)}, fd, discrepancy_limit)
				if ms > max_score:
					max_score, best_grid = ms, bg
		return max_score, best_grid	
	def postprocess(self, max_score, best_grid): 
		change = True
		while change:
			change = False
			for i, num in self.candgen.rewards:
				_, length = self.clues[i][num]
				
				# word = get_string(grid, *self.num2pos[num], i, length)
				# best_word, max_reward = None, 0
				# for w, (r, _, _) in self.candgen.rewards[i, num].items():
				# 	if max_reward < r:
				# 		best_word, max_reward = w, r
				# if best_word != word: pass

				ngrid = fill(best_grid, i, *self.num2pos[num], "*"*length, True)
				ms, bg = self._lds(0, ngrid)
				if ms > max_score:
					# print("Postprocess improvement: %s->%s"%(max_score, ms))
					# print(grid2string(best_grid))
					# print("="*20)
					# print(grid2string(bg))
					max_score, best_grid = ms, bg
					change =True
					
		return max_score, np.array(best_grid, dtype="S")
	def search(self, grid, time_limit=0, fdeb=None, answers=None, time_scale=60):
		return self.search_autostop(grid, time_limit, fdeb, answers, 0, False, time_scale)
	def search_autostop(self, grid, time_limit=0, fdeb=None, answers=None, tolerance=.5, verbose=True, time_scale=60, max_rt=None):
		t0 = time.time()
		self.time_limit = t0+time_limit if time_limit else 1e11
		self.fdeb = fdeb
		self.action_buf = OrderedDict()
		self.state_buf = {}
		self.n_states = 0

		self.timings = {"est": [0, 0], "act": [0, 0], "\tclueG": [0, 0], "\tvocabG":[0, 0], "\tsort":[0, 0]}
		if fdeb:
			self.answers = answers
		if answers is not None:
			agrid = grid.copy()
			for i in range(len(answers)):
				for num, ans in answers[i].items():
					agrid = fill(agrid, i, *self.num2pos[num]. ans, False)
		else: agrid = None

		max_score, best_grid = -1e9, None
		self.n_iters, ast = 0, 0
		max_t_iter = 1 # save at least 1 secs for postprocess
		last_update = time.time()
		solutions = {}
		scores = []
		fdcounts = []
		best_trajectory = None
		try:
			rt = self.time_limit - time.time()
			while time.time()+max_t_iter < self.time_limit:
				t_start = time.time()
				self.action_list = []
				ms, bg = self._select_and_expand(0, grid.copy(), set())
				scores.append(ms)
				fdcounts.append(sum(int(act[3] > 0) for act in self.action_list))
				self.candgen.update_rewards(ms, bg, self.num2pos)
				ss = self.get_state_hash(bg, set())
				if ss not in solutions: solutions[ss] = (ms, bg)
				rt = self.time_limit - time.time()
				max_t_iter = max(max_t_iter, self.time_limit - rt - t_start)
				self.n_iters += 1
				r_time = time.time()-t0
				sys.stdout.write("Iteration %6d (%.4f iters/s), remaining: %.2fs\r"%(self.n_iters, self.n_iters/r_time, time_limit-r_time))
				sys.stdout.flush()
				if ms > max_score:
					max_score, best_grid = ms, bg
					best_trajectory = self.action_list
					last_update = time.time()
					if verbose:
						best_grid = np.array(best_grid, dtype="S")
						n_blanks = (best_grid == blank_sign).sum()
						n_total = (best_grid != block_sign).sum()
						n_filled = n_total - n_blanks
						print("new best score %f at %.2f"%(max_score, last_update-t0))
						print(grid2string(best_grid))
						print("Completeness: %d/%d=%.2f%%"%(n_filled, n_total, n_filled/n_total*100))
						
					if fdeb and agrid is not None:
						tot_words = len(answers[0]) + len(answers[1])
						n_words, n_letters = evaluate(bg, agrid, self.num2pos, answers)
						if verbose:
							tot_letters = int((agrid != b".").sum())
							r_words = n_words / tot_words
							r_letters = n_letters / tot_letters
							print("Acc words: %d/%d=%.2f%%"%(n_words, tot_words, r_words*100))
							print("Acc letters: %d/%d=%.2f%%"%(n_letters, tot_letters, r_letters*100))
						if n_words == tot_words:
							print("DEBUG: the answer grid found, break")
							break
					# if (best_grid == blank_sign).sum() == 0:
					# 	print("The perfect grid is found, break")
					# 	break
					ast = self.autostop_tolerance(tolerance*time_scale, max_score)
				elif tolerance and time.time() - last_update > ast and int(rt/time_scale) > int((rt-max_t_iter)/time_scale):
					print("No better grids found in %.2f seconds"%(time.time() - last_update), end="...")
					if max_rt is not None and rt-max_t_iter > max_rt: 
						print("Remaining time (%.1f) less than max_rt (%.1f), resume"%(rt, max_rt))
					else: 
						print("Break")
						break

				if self.candgen.check_async(): # if the async candidates are generated
					print('Time: %.2f, async candidate generation is done, update and clear the action buf'%(time.time()-t0))
					self.candgen.update_cands() # update the async candidates
		except KeyboardInterrupt:
			print("Interupted manually")
		print("MCTS is done, %s iterations with %d unique solutions"%(self.n_iters, len(solutions)))
		print("Remaining time: %.4f. Module timings"%rt)
		for k, (tot_delta_t, n_times) in self.timings.items():
			print("%s=%.2f/%d=%.3e"%(k, tot_delta_t, n_times, tot_delta_t/(n_times+1e-6)))
		#print("Postprocessing high-reward solutions...")
		if self.doPost:
			t_pstart = time.time()
			old_max = max_score
			max_t_iter = cnt = 0
			solutions = sorted(solutions.values(), key=lambda x:x[0])
			while solutions:
				ms, bg = solutions.pop()
				t_start = time.time()
				ms, bg = self.postprocess(ms, bg)
				max_t_iter = max(max_t_iter, time.time()-t_start)
				cnt += 1
				if ms > max_score:
					max_score, best_grid = ms, bg
				rt = self.time_limit - time.time()
				if rt < 0 or math.floor(rt/time_scale) > math.floor((rt-max_t_iter)/time_scale): break
			print("Postprocessed %d solutions in %.4f seconds, improve reward from %s to %s"%\
				(cnt, time.time()-t_pstart, old_max, max_score))
		del solutions
		return max_score, np.array(best_grid, dtype="S"), scores, fdcounts, best_trajectory
	def autostop_tolerance(self, base, score): return base

class StateHeap:
	def __init__ (self):
		self.pool = [] # state pool
		self.idle = [] # idle indices
		self.heap = [] # maximum root heap
	def push(self, state):
		if self.idle:
			idx = self.idle.pop()
		else:
			idx = len(self.pool)
			self.pool.append(None)
			
		self.pool[idx] = state
		self.heappush(idx)
	def pop(self):
		if self.heap:
			idx = self.heappop()
			self.idle.append(idx)
			return self.pool[idx]
		else:
			return None

	def siftdown(self, startpos, pos):
		newitem = self.heap[pos]
		while pos > startpos:
			parentpos = (pos - 1) >> 1
			parent = self.heap[parentpos]
			if self.pool[newitem][0] > self.pool[parent][0]:
				self.heap[pos] = parent
				pos = parentpos
				continue
			break
		self.heap[pos] = newitem

	def heappush(self, idx):
		self.heap.append(idx)
		self.siftdown(0, len(self.heap)-1)

	def siftup(self, pos):
		endpos = len(self.heap)
		startpos = pos
		newitem = self.heap[pos]
		childpos = 2*pos + 1    # leftmost child position
		while childpos < endpos:
			# Set childpos to index of smaller child.
			rightpos = childpos + 1
			if rightpos < endpos and not self.pool[self.heap[childpos]][0] > self.pool[self.heap[rightpos]][0]:
				childpos = rightpos
			# Move the smaller child up.
			self.heap[pos] = self.heap[childpos]
			pos = childpos
			childpos = 2*pos + 1
		# The leaf at pos is empty now.  Put newitem there, and bubble it up
		# to its final resting place (by sifting its parents down).
		self.heap[pos] = newitem
		self.siftdown(startpos, pos)
	
	def heappop(self):
		lastelt = self.heap.pop()
		if self.heap:
			returnitem = self.heap[0]
			self.heap[0] = lastelt
			self.siftup(0)
			return returnitem
		return lastelt

class Astar(MCTS):
	@property
	def greedy_factor(self): return 1
	def heuristic(self, grid):
		rw, unfilled = self.est_grid(grid)
		h = 0
		for i, num, word in unfilled:
			cands = self.candgen.generate(num, i, word)
			if cands:
				_, s = cands[0]
				h += s
		return rw, h / self.cp_size, unfilled
	def get_childrens(self, grid, unfilled):
		self.timings["act"][1] += 1
		self.timings["act"][0] -= time.time()
		childrens = []
		for i, num, string in unfilled: 
			cands = self.candgen.generate(num, i, string, timings=self.timings)
			if len(cands):
				w, _ = cands[0]
				g = fill(grid, i, *self.num2pos[num], w)
				s, h, uf = self.heuristic(g)
				childrens.append([s+self.greedy_factor * h, s, num, i, w, g, uf]) 
		childrens.sort(key=lambda x:-x[0])
		self.timings["act"][0] += time.time()
		return childrens
	def search_autostop(self, grid, time_limit=0, fdeb=None, answers=None, tolerance=.5, verbose=True, time_scale=60, **kwargs):
		t0 = time.time()
		self.time_limit = t0+time_limit if time_limit else 1e11
		self.action_buf = {}
		self.state_buf = {}
		self.n_states = 0
		self.max_score = -1e9
		self.best_grid = None
		self.timings = {"est": [0, 0], "act": [0, 0], "\tclueG": [0, 0], "\tvocabG":[0, 0], "\tsort":[0, 0]}
		if fdeb:
			self.answers = answers
		if answers is not None:
			agrid = grid.copy()
			for i in range(len(answers)):
				for num, ans in answers[i].items():
					agrid = fill(agrid, i, *self.num2pos[num], ans, False)
		else: agrid = None

		self.n_iters, ast = 0, 0
		max_t_iter = 0
		last_update = time.time()
		solutions = {}
		scores = []
		fdcounts = []
		best_trajectory = self.action_list = []

		state_heap = StateHeap()
		max_score, unfilled = self.est_grid(grid)
		best_grid = grid
		scores.append(max_score)
		fdcounts.append(0)
		last_t = time.time()
		try:
			while last_t+max_t_iter < self.time_limit:
				self.n_states += 1
				childrens = self.get_childrens(grid, unfilled)
				for c in childrens:
					f, s, num, i, w, g, uf = c
					ss = self.get_state_hash(g, set())
					if ss not in solutions: 
						solutions[ss] = (s, g)
						if uf:
							state_heap.push(c)
						if s > max_score:
							max_score, best_grid = s, g
							self.max_score, self.best_grid = max_score, best_grid
							last_update = time.time()

					rt = self.time_limit - time.time()
					if tolerance and time.time() - last_update > ast and int(rt/time_scale) > int((rt-max_t_iter)/time_scale):
						print("No better grids found in %.2f seconds, break"%(time.time() - last_update))
						break
				if not state_heap.heap: break
				f, s, num, i, w, grid, unfilled = state_heap.pop()
				max_t_iter = max(max_t_iter, time.time() - last_t)
				last_t = time.time()
		except KeyboardInterrupt:
			print("Interupted manually")
		print("Remaining time: %.4f. Module timings"%rt)
		for k, (tot_delta_t, n_times) in self.timings.items():
			print("%s=%.2f/%d=%.3e"%(k, tot_delta_t, n_times, tot_delta_t/(n_times+1e-6)))

		t_pstart = time.time()
		old_max = max_score
		max_t_iter = cnt = 0
		solutions = sorted(solutions.values(), key=lambda x:x[0])
		while solutions:
			ms, bg = solutions.pop()
			t_start = time.time()
			ms, bg = self.postprocess(ms, bg)
			max_t_iter = max(max_t_iter, time.time()-t_start)
			cnt += 1
			if ms > max_score:
				max_score, best_grid = ms, bg
			rt = self.time_limit - time.time()
			if rt < 0 or math.floor(rt/time_scale) > math.floor((rt-max_t_iter)/time_scale): break
		print("Postprocessed %d solutions in %.4f seconds, improve reward from %s to %s"%\
			(cnt, time.time()-t_pstart, old_max, max_score))
		print("%d solutions found"%len(solutions))
		del solutions
		return max_score, np.array(best_grid, dtype="S"), scores, fdcounts, best_trajectory

class LDS(MCTS):
	def search_autostop(self, grid, time_limit=0, fdeb=None, answers=None, tolerance=.5, verbose=True, time_scale=60, **kwargs):
		t0 = time.time()
		self.time_limit = t0+time_limit if time_limit else 1e11
		self.action_buf = {}
		self.state_buf = {}
		self.n_states = 0
		self.max_score = -1e9
		self.best_grid = None
		self.timings = {"est": [0, 0], "act": [0, 0], "\tclueG": [0, 0], "\tvocabG":[0, 0], "\tsort":[0, 0]}
		if fdeb:
			self.answers = answers
		if answers is not None:
			agrid = grid.copy()
			for i in range(len(answers)):
				for num, ans in answers[i].items():
					agrid = fill(agrid, i, *self.num2pos[num]. ans, False)
		else: agrid = None

		max_score, best_grid = -1e9, None
		self.n_iters, ast = 0, 0
		max_t_iter = 1 # save at least 1 secs for postprocess
		last_update = time.time()
		solutions = {}
		scores = []
		fdcounts = []
		best_trajectory = self.action_list = []
		try:
			ds = 0
			while time.time()+max_t_iter < self.time_limit:
				ms, bg = self._lds(0, grid, discrepancy_limit=ds)
				ss = self.get_state_hash(bg, set())
				if ss not in solutions: solutions[ss] = (ms, bg)
				rt = self.time_limit - time.time()
				if ms > max_score:
					max_score, best_grid = ms, bg
					best_trajectory = self.action_list
					last_update = time.time()
				elif tolerance and time.time() - last_update > ast and int(rt/time_scale) > int((rt-max_t_iter)/time_scale):
					print("No better grids found in %.2f seconds, break"%(time.time() - last_update))
					break
				fdcounts.append(0)
				ds += 1
		except KeyboardInterrupt:
			print("Interupted manually")
		print("Remaining time: %.4f. Module timings"%rt)
		for k, (tot_delta_t, n_times) in self.timings.items():
			print("%s=%.2f/%d=%.3e"%(k, tot_delta_t, n_times, tot_delta_t/(n_times+1e-6)))

		t_pstart = time.time()
		old_max = max_score
		max_t_iter = cnt = 0
		solutions = sorted(solutions.values(), key=lambda x:x[0])
		while solutions:
			ms, bg = solutions.pop()
			t_start = time.time()
			ms, bg = self.postprocess(ms, bg)
			max_t_iter = max(max_t_iter, time.time()-t_start)
			cnt += 1
			if ms > max_score:
				max_score, best_grid = ms, bg
			rt = self.time_limit - time.time()
			if rt < 0 or math.floor(rt/time_scale) > math.floor((rt-max_t_iter)/time_scale): break
		print("Postprocessed %d solutions in %.4f seconds, improve reward from %s to %s"%\
			(cnt, time.time()-t_pstart, old_max, max_score))
		del solutions
		return max_score, np.array(best_grid, dtype="S"), scores, fdcounts, best_trajectory

class DrFill(LDS):
	def heuristic(self, grid):
		rw, unfilled = self.est_grid(grid)
		h = 0
		for i, num, word in unfilled:
			cands = self.candgen.generate(num, i, word)
			if cands:
				_, s = cands[0]
				h += s
		return rw + h / self.cp_size
	def get_actions(self, grid, blacklist, fold, ss=None, mode=0): # mode0: selection; mode1: simulation
		score, unfilled = self.est_grid(grid)
		self.timings["act"][1] += 1
		self.timings["act"][0] -= time.time()
		actions = []
		for i, num, string in unfilled: 
			cands = self.candgen.generate(num, i, string, timings=self.timings)
			ff = fold
			while len(cands) > ff and (i, num, cands[ff][0]) in blacklist: ff += 1
			if ff < len(cands):
				w, s = cands[ff]
				g = fill(grid, i, *self.num2pos[num], w)
				h = self.heuristic(g)
				if len(cands) > ff+1:
					w_, s_ = cands[ff+1]
					g_ = fill(grid, i, *self.num2pos[num], w_)
					second_largest_score = self.heuristic(g_)
				else:
					second_largest_score = self.heuristic(grid)
				
				actions.append([second_largest_score-h, num, i, w, g, blacklist, 0]) 
		self.timings["act"][0] += time.time()
		ranked_actions = self.rank_actions(grid, actions)
		return ranked_actions, score

class MCTS_DG(MCTS): # for training data generation
	def generate_data(self, grid, agrid, time_limit, answers):
		self.time_limit = time.time()+time_limit if time_limit else 1e11
		self.action_buf = {}
		self.state_buf = {}
		self.n_states = 0
		self.max_score = -1e9
		self.best_grid = None
		self.answers = answers
		max_score, best_grid = -1e9, None
		n_iters = 0
		# get grids
		hgrids = {concate_bytes(grid), concate_bytes(agrid)} # add init grid and answer grid into the set
		self.timings = {"est": [0, 0], "act": [0, 0]}
		while time.time() < self.time_limit:
			ms, bg = self._select_and_expand(0, grid, set())
			ms, bg = ms, np.array(bg, dtype="S")
			if ms > max_score:
				max_score = ms
				best_grid = bg
			hgrids.add(concate_bytes(bg))
			n_iters += 1
			self.candgen.update_rewards(ms, bg, self.num2pos)
		print("Timings")
		for k, (tot_delta_t, n_times) in self.timings.items():
			print("%s=%.2f/%d=%.3e"%(k, tot_delta_t, n_times, tot_delta_t/n_times))
		print("MCTS is done, %s iterations"%n_iters)
		print("%d unique grids" % len(hgrids))
		return max_score, best_grid, hgrids