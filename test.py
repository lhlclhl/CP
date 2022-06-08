
import random, math, os, time, json, traceback, sys, argparse
import numpy as np, unidecode, h5py, inspect
from collections import defaultdict
from itertools import count
from os.path import exists, join, split
from functools import lru_cache
import cps.puz_utils
from cps.puz_utils import *
from cps.search import *


pzldir = "data/puzzles/"
outdir = "outputs/"
detdir = join(outdir, "details_full")
ovadir = join(outdir, "overall_full")
imddir = "intermediate/puz_values"
dbgdir = "intermediate/debug_info"
bufdir = "intermediate/candidates"
rwddir = "intermediate/reward_feats"
if not exists(imddir): os.makedirs(imddir)
if not exists(dbgdir): os.makedirs(dbgdir)
if not exists(bufdir): os.makedirs(bufdir)
#sys.setrecursionlimit(10000)
method_list = dict(
	m for m in inspect.getmembers(sys.modules[__name__]) 
	if inspect.isclass(m[1]) and issubclass(m[1], CPSolver)
)

def run_test(fn, time_limits=100, cand_limit=50, methods=["MCTS"], start_from=None, end_with=None, \
append=False, output=None, args=None, debug=False, norm=False, candbuff=0, write_golden=0, n_repeats=5):
	for i in range(len(methods)):
		if methods[i] in method_list:
			methods[i] = method_list[methods[i]](args, 
			clue_dbname="cluedblfs",
			clue_dir="data/clues_before_2020-09-26/",
			vocab_path="data/dictionaries/vocab_2020-09-26.txt"
		)
		else: raise Exception("Unregistered method", methods[i])
	skip = start_from is not None
	if not exists(detdir): os.makedirs(detdir)
	if not exists(ovadir): os.makedirs(ovadir)
	ofn = output if output else split(fn)[-1]
	
	mfil = set()
	if append == 2: # filter out the existing results
		with open(join(ovadir, ofn), encoding="utf-8") as fin:
			fin.readline()
			for line in fin:
				lln = line.strip("\r\n").split("\t")
				mfil.add((lln[0], lln[1], int(lln[2])))

	# load puzzles
	puzzles = []
	with open(fn, encoding="utf-8") as fin:
		for line in fin:
			puzzle = json.loads(line)
			pzlid = puzzle["date"]
			if pzlid == start_from: skip = False
			if skip: continue
			if pzlid == end_with: skip = True
			try:
				grid, num2pos, pos2num, clues, answers, agrid = construct_grid(puzzle)
			except Exception as e: 
				print("Special Grid")
				continue
			if agrid.dtype != np.dtype("S1"):
				print("Multi-letter Grid")
				continue
			puzzles.append((puzzle["title"], grid, num2pos, pos2num, clues, answers, agrid))
	print("Number of test puzzles", len(puzzles))

	with open(join(ovadir, ofn), "a" if append else "w", encoding="utf-8") as fout, \
	open(join(detdir, ofn), "a" if append else "w", encoding="utf-8") as fdet:
		if not append:
			fout.write("puzzle\tmethod\ttime\twords\tletters\tstates\tscores\tascore\tpreptime\tsearchtime\n")
		
		for nr in range(n_repeats): # to avoid the influence of es buffer, do not run one puzzle repeatedly
			print("Round %d start"%(nr+1))

			for method in methods:
				print("Testing method %s"%(str(method)))

				for pid, (tit, grid, num2pos, pos2num, clues, answers, agrid) in enumerate(puzzles):
					if (tit, str(method), time_limits) in mfil:
						print("this setting has already run, skip")
						continue
					print(grid.shape)
					print(grid2string(agrid))
					tot_words = len(answers[0]) + len(answers[1])
					tot_letters = (agrid != b".").sum()

					t0 = time.time() # start timing
					method.initialize(tit, agrid, clues, num2pos, answers=answers, async_start=True)
					t_prep = time.time()
					print("Method %s preparation time: %.2f"%(str(method), t_prep-t0))
					print("Round %d/%d\tPuzzle %d/%d"%(nr+1, n_repeats, pid+1, len(puzzles)))
					rt = time_limits-t_prep+t0
					print("Start solving golden puzzle %s in %.2f seconds with %s"%(tit, rt, method))
					try:
						if debug:
							dbsdir = join(dbgdir, str(method))
							if not exists(dbsdir): os.makedirs(dbsdir)
							fdeb = open(join(dbsdir, "%s.txt"%(tit.replace("/", "-"))), "a" if append else "w", encoding="utf-8")
						ms, bg, scores, fdcounts, best_trajectory = method.search(grid, rt, fdeb=fdeb if debug else None, answers = answers if debug else None)
						if debug:
							fdeb.close()
					except Exception: 
						traceback.print_exc()
						continue
					t_search = time.time()
					print("-"*20)
					print(grid2string(bg))
					print("runtime: %.2f"%(t_search-t0))
					print("max score", ms)
					n_words, n_letters = evaluate(bg, agrid, num2pos, answers)
					r_words = n_words / tot_words
					r_letters = n_letters / tot_letters
					asc, _ = method.est_grid(agrid)
					print("ans score", asc)
					avg_fdcounts = sum(fdcounts)/len(fdcounts)
					best_fd = sum(int(fd > 0) for n, i, w, fd in best_trajectory)

					fout.write(f"{tit}\t{str(method)}\t{time_limits}\t{n_words}/{tot_words}={r_words:.3f}\t{n_letters}/{tot_letters}={r_letters:.3f}\t{method.n_states}\t{ms}\t{asc}\t{t_prep-t0:.2f}\t{t_search-t0:.2f}\n")
					fout.flush()
					fdet.write(json.dumps({
						"title": tit,
						"grid": [[x.decode() for x in row] for row in agrid],
						"output": [[x.decode() for x in row] for row in bg],#[list(concate_bytes(row)) for row in bg],
						"algorthm": str(method),
						"time": t_search-t0,
						"clues": clues,
						"answers": answers,
						"num2pos": num2pos.tolist(),
						"r_words": r_words,
						"r_letters": r_letters,
						"candidates": method.candidates,
						"n_states": method.n_states,
						"max_score": ms,
						"ans_score": asc,
						"score_iter": scores,
						"fdcounts_iter": fdcounts,
						"best_trajectory": best_trajectory,
					}, ensure_ascii=False)+"\n")
					fdet.flush()
					method.clear()
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode',  type=str, default="test")
	parser.add_argument('--methods',  type=str, default="MCTS")
	parser.add_argument('--time_limits',  type=str, default="100")
	parser.add_argument('--train_file',  type=str, default=join(pzldir, "nyt.shuffle.txt"))
	parser.add_argument('--test_file',  type=str, default=join(pzldir, "nyt.new.ra.txt"))
	parser.add_argument('--append', default=0, type=int)
	parser.add_argument('--output', type=str, default=None)
	parser.add_argument('--debug', default=False, action='store_true')
	parser.add_argument('--norm', default=False, action='store_true')
	parser.add_argument('--start_from', type=str, default=None)
	parser.add_argument('--end_with',  type=str, default=None)
	parser.add_argument('--candbuff',  type=int, default=1)
	parser.add_argument('--cand_limit',  type=int, default=50)
	parser.add_argument('--write_golden',  type=int, default=0)
	parser.add_argument('--n_repeats',  type=int, default=5)
	parser.add_argument('--sleep',  type=float, default=0)
	pargs, unkargs = parser.parse_known_args()
	if pargs.output is None: pargs.output = pargs.methods + ".txt"
	if pargs.sleep:
		print("sleeping for %.2f secs"%pargs.sleep)
		time.sleep(pargs.sleep)
	run_test(pargs.test_file, 
		time_limits=int(pargs.time_limits), 
		cand_limit=pargs.cand_limit,
		methods=pargs.methods.split(","), 
		append=pargs.append, 
		output=pargs.output, 
		args=unkargs,
		start_from=pargs.start_from,
		end_with=pargs.end_with,
		debug=pargs.debug,
		norm=pargs.norm,
		candbuff=pargs.candbuff,
		write_golden=pargs.write_golden,
		n_repeats=pargs.n_repeats
	)