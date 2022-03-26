import argparse, sys, inspect, json
from cps.reward import *

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--method',  type=str, default="RNRewarder")
	parser.add_argument('--loss',  type=str, default=None)
	parser.add_argument('--margin',  type=int, default=None)
	parser.add_argument('--temp',  type=int, default=None)
	parser.add_argument('--n_nw',  type=int, default=None)
	parser.add_argument('--zero_init',  type=int, default=None)
	parser.add_argument('--feat_norm',  type=int, default=None)
	parser.add_argument('--n_negs',  type=int, default=None)
	parser.add_argument('--fm',  type=int, default=None)
	parser.add_argument('--l2',  type=int, default=None)
	parser.add_argument('--mask_ratio',  type=float, default=None)
	parser.add_argument('--r_nw',  type=float, default=None)
	parser.add_argument('--lr',  type=float, default=None)
	parser.add_argument('--epsilon',  type=float, default=None)
	parser.add_argument('--smooth',  type=float, default=None)
	parser.add_argument('--output',  type=str, default="reward_weights.txt")
	pargs, unkargs = parser.parse_known_args()
	if pargs.method in method_list:
		kwargs = {
			k:v for k, v in pargs._get_kwargs() if k != "method" and v is not None
		}
		rewarder = method_list[pargs.method](**kwargs)
		rewarder.train()
		rewarder.load()
		weights = {i:w for i, w in enumerate(rewarder._pmodel.get_weights()[0].flatten().tolist()) if w != 0}
		if len(weights) <= 100:
			with open(pargs.output, "a", encoding="utf-8") as fout:
				fout.write("%s\t%s\n"%(str(rewarder), json.dumps(weights)))
			
			print("weights", weights)
	else:
		raise Exception("Unregistered method: %s"%pargs.method)
