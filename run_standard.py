import os, sys, time

methods = ["MCTS"] # method class
append = 0
settings = [(1, 100)] # (repetitions, time_limit)
outfile = "standard.txt"
args =  "--test_file data/puzzles/nyt.new.ra.txt"#"--end_with 11/7/2020"
#time.sleep(16000)
for n_repeats, time_limit in settings:
	if type(n_repeats) != list: n_repeats = [n_repeats] * len(methods)
	while sum(n_repeats) > 0:
		for i, method in enumerate(methods):
			if n_repeats[i] == 0: continue
			#if method == "MCTS_U2" and time_limit == 100: continue
			t0 = time.time()
			os.system("python3 -W ignore test.py --methods %s --n_repeats 1 --time_limits %d --output %s --append %s %s %s"%\
				(method, time_limit, outfile, append, args, " ".join(sys.argv[1:])))
			if time.time()-t0 < 180:
				print("exit with error, quit")
				sys.exit()
			append = 1
			n_repeats[i] -= 1
