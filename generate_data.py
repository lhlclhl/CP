import random, math, os, time, json, traceback, sys, argparse
from datetime import datetime
import numpy as np, unidecode, h5py
from collections import defaultdict
from itertools import count
from os.path import exists, join, split
from functools import lru_cache
import cps.puz_utils
from cps.puz_utils import *
from cps.search import *


moddir = "models/value_net"
pzldir = "data/puzzles/"
outdir = "outputs/"
detdir = join(outdir, "details")
ovadir = join(outdir, "overall")
imddir = "intermediate/puz_values"
dbgdir = "intermediate/saved_grids"
svwdir = "intermediate/saved_words"
bufdir = "intermediate/candidates"
rwddir = "intermediate/reward_feats"
if not exists(imddir): os.makedirs(imddir)
if not exists(moddir): os.makedirs(moddir)
if not exists(dbgdir): os.makedirs(dbgdir)
if not exists(svwdir): os.makedirs(svwdir)
if not exists(bufdir): os.makedirs(bufdir)
#sys.setrecursionlimit(10000)
from cps.mcts import method_list

def run_DG(fn, time_limit=100, cand_limit=50, method="MCTS_DG", start_from=None, end_with=None, \
append=False, output=None, args=None, date_after="6/18/2020", **kwargs):
	if method in method_list:
		method = method_list[method](args, clue_dbname="cluedb200618", \
			clue_dir="data/clues_before_2020-06-18", \
			vocab_path= "data/dictionaries/vocab_2020-06-18.txt", \
			stringfeat_path= "data/dictionaries/stringfeats_200618.txt", **kwargs)
	else: raise Exception("Unregistered method", method)
	skip = start_from is not None
	if not exists(ovadir): os.makedirs(ovadir)
	ofn = output if output else split(fn)[-1]
	
	mfil = set()
	if append == 2: # filter out the existing results
		with open(join(ovadir, ofn), encoding="utf-8") as fin:
			fin.readline()
			for line in fin:
				lln = line.strip("\r\n").split("\t")
				mfil.add((lln[0], lln[1], int(lln[2])))

	with open(fn, encoding="utf-8") as fin, \
	open(join(ovadir, ofn), "a" if append else "w", encoding="utf-8") as fout:
		if not append:
			fout.write("puzzle\tmethod\ttime\twords\tletters\tstates\tscores\tascore\truntime\n")
		for line in fin:
			puzzle = json.loads(line)
			date = puzzle["date"]
			dt = datetime.strptime(date, "%m/%d/%Y")
			if dt <= datetime.strptime(date_after, "%m/%d/%Y"): 
				print("skip date", date)
				continue
			pzlid = date.replace("/", "-")
			if puzzle["date"] == start_from: skip = False
			if skip: continue
			if puzzle["date"] == end_with: skip = True
			
			featdir = join(rwddir, str(method))
			if not exists(featdir): os.makedirs(featdir)
			gfeat_file = join(featdir, "%s.h5"%pzlid)
			if append and exists(gfeat_file):
				print("append skip", pzlid)
				continue
			
			try:
				grid, num2pos, pos2num, clues, answers, agrid = construct_grid(puzzle)
			except Exception as e: 
				print("Special Grid")
				continue
			if agrid.dtype != np.dtype("S1"):
				print("Multi-letter Grid")
				continue
			print(grid.shape)
			print(grid2string(agrid))

			tot_words = len(answers[0]) + len(answers[1])
			tot_letters = (agrid != b".").sum()

			if (puzzle["title"], str(method), time_limit) in mfil:
				print("this setting has already run, skip")
				continue

			dbsdir = join(dbgdir, str(method))
			if not exists(dbsdir): os.makedirs(dbsdir)
			datafile = join(dbsdir, "%s.h5"%(pzlid))
			worddir = join(svwdir, str(method))
			if not exists(worddir): os.makedirs(worddir)
			wordfile = join(worddir, "%s.txt"%(pzlid)) # candidate word file for each word grouped by clue-aware and clue-free retrieval
			featfile = join(worddir, "%s.feat"%(pzlid)) # feature file for each candidate word to each clue
			if not exists(datafile) or not exists(wordfile):
				# initialize 
				t0 = time.time()
				method.initialize(puzzle['date'], agrid, clues, num2pos, answers=answers)
				print("Method %s preparation time: %.2f"%(str(method), time.time()-t0))
				print("Start solving golden puzzle %s in %d seconds with %s"%(puzzle["title"], time_limit, method))

				# run search
				t0 = time.time()
				try:
					ms, bg, hgrids = method.generate_data(grid, agrid, time_limit, answers)
				except Exception: 
					traceback.print_exc()
					continue
				delta_t = time.time()-t0
				print("-"*20)
				print(grid2string(bg))
				print("runtime: %.2f"%(delta_t))
				print("max score", ms)
				asc, _ = method.est_grid(agrid)
				print("ans score", asc)
				
				n_words, n_letters = evaluate(bg, agrid, num2pos, answers)
				r_words = n_words / tot_words
				r_letters = n_letters / tot_letters

				fout.write(f"{puzzle['title']}\t{str(method)}\t{time_limit}\t{n_words}/{tot_words}={r_words:.3f}\t{n_letters}/{tot_letters}={r_letters:.3f}\t{method.n_states}\t{ms}\t{asc}\t{delta_t:.2f}\n")
				fout.flush()

				# construct puzzle data
				grids = np.zeros((len(hgrids),) + grid.shape, dtype=grid.dtype)
				Ys = np.zeros((len(hgrids), 4), dtype="int32") # n_words, tot_words, n_letters, tot_letters
				for k, g in enumerate(hgrids):
					for i in range(grid.shape[0]):
						for j in range(grid.shape[1]):
							grids[k][i][j] = g[i*grid.shape[1]+j]

					n_words, n_letters = evaluate(grids[k], agrid, num2pos, answers)
					Ys[k] = [n_words, tot_words, n_letters, tot_letters]
				masked = np.array(method.masked_cands, dtype='int32')

				with h5py.File(datafile, "w") as dfile:
					dfile.create_dataset("grids", data=grids)
					dfile.create_dataset("Ys", data=Ys)
					dfile.create_dataset("masked", data=masked)

				# construct word data
				candgen = method.candgen
				vocab_cands = defaultdict(list)
				for (i, num, ptn), cands in candgen.cands_from_vocab.items():
					vocab_cands[i, num].append([w for w, r in cands])
				with open(wordfile, "w", encoding='utf-8') as fwd, \
				open(featfile, "w", encoding="utf-8") as fft:
					for i in range(len(clues)):
						for num, (clue, l) in clues[i].items():
							x, y = num2pos[num]
							ans = answers[i][num]
							cands = candgen.cands_from_cluedb[i][num]
							# words = [(w, cands[w][0].tolist() if w in cands else None) \
							# 	for w in method.candgen.rewards[i, num]]
							fwd.write("%d,%d\t%s\t%s\t%s\t%s\n"%(i, num, clue, ans, \
								json.dumps([w for w in cands], ensure_ascii=False),\
								json.dumps(vocab_cands[i, num], ensure_ascii=False)))
							for w, r in candgen.rewards[i, num].items():
								mscore = cands.get(w, [candgen.drv, None])[0]
								feat = candgen.make_feature(w, clue, candgen.clue_pos[i, num], mscore)
								fft.write("%d,%d\t%s\t%s\t%s\n"%(i, num, w, json.dumps(r), \
									json.dumps(feat.tolist())))
				method.clear()
			else: 
				with h5py.File(datafile, "r") as dfile:
					grids = dfile["grids"][:]
					Ys = dfile["Ys"][:]
					masked = dfile["masked"][:].tolist()

				# initialize 
				t0 = time.time()
				method.initialize(puzzle['date'], agrid, clues, num2pos, answers=answers, masked_cands=masked)
				print("Method %s preparation time: %.2f"%(str(method), time.time()-t0))

			# making features
			# X: linear feature
			# Y: (n_words+n_letters/tot_letters)/(tot_words+1)
			candgen = method.candgen
			X_feats = np.zeros((len(grids), len(candgen.features)), dtype="float32")
			Y_scores = np.zeros((len(grids)), dtype="float32")
			for k, (ngrid, (n_words, tot_words, n_letters, tot_letters)) in enumerate(zip(grids, Ys)):
				samples = []
				for i in range(len(clues)):
					for num, (clue, l) in clues[i].items():
						x, y = num2pos[num]
						word = get_string(ngrid, x, y, i, l)
						samples.append((i, num, word))
				feat = candgen.make_features(samples)
				X_feats[k] = feat.mean(axis=0)
				Y_scores[k] = (n_words+n_letters/tot_letters)/(tot_words+1)
			print("feature constructed", X_feats.shape)
			with h5py.File(gfeat_file, "w") as dfile:
				dfile.create_dataset("X_feats", data=X_feats)
				dfile.create_dataset("Y_scores", data=Y_scores)
			del X_feats, Y_scores

			
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode',  type=str, default="test")
	parser.add_argument('--methods',  type=str, default="MCTS_DG")
	parser.add_argument('--time_limits',  type=str, default="100")
	parser.add_argument('--train_file',  type=str, default=join(pzldir, "nyt.valid.txt"))
	parser.add_argument('--append', default=0, type=int)
	parser.add_argument('--output', type=str, default=None)
	parser.add_argument('--start_from', type=str, default=None)
	parser.add_argument('--end_with',  type=str, default=None)
	parser.add_argument('--cand_limit',  type=int, default=50)
	parser.add_argument('--mask_ratio',  type=float, default=0)
	parser.add_argument('--sleep',  type=float, default=0)
	pargs, unkargs = parser.parse_known_args()
	if pargs.output is None: pargs.output = pargs.methods + ".txt"
	if pargs.sleep:
		print("sleeping for %.2f secs"%pargs.sleep)
		time.sleep(pargs.sleep)
	run_DG(pargs.train_file, 
		time_limit=[int(tl) for tl in pargs.time_limits.split(",")][0], 
		method=pargs.methods.split(",")[0], 
		append=pargs.append, 
		output=pargs.output, 
		args=unkargs,
		start_from=pargs.start_from,
		end_with=pargs.end_with,
		# **kwargs
		mask_ratio=pargs.mask_ratio,
		cand_limit=pargs.cand_limit,
	)