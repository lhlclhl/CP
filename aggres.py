import os, math, sys
from os.path import join
from operator import itemgetter
from collections import defaultdict
mfa = {
	"words": (0, "%.2f", "perc"),
	"letters": (1, "%.2f", "perc"),
	"scores": (2, "%.4f", "float"),
	"preptime": (3, "%.1f", "float"),
	"searchtime": (4, "%.1f", "float"),
	"runtime": (5, "%.1f", "float"),
	"states": (6, "%d", "int"),
	"Repeats": (7, "%d", None),
	"ascore": (8, "%.4f", "float"),
}
def aggregate_results(ouf="res.txt", infs="outputs/overall", rdir="",\
reduce_field = "puzzle", spec_fields = ["time", "method"], bl=set()):
	print("blacklist methods", list(bl))
	results = defaultdict(lambda :defaultdict(list))
	mtc_fields = set()
	if type(infs) == str:
		print("Aggregate all the files in dir", infs)
		rdir = infs
		infs = os.listdir(rdir)
		ouf = join(rdir, ouf)
	for inf in infs:
		if join(rdir, inf) == ouf or inf.startswith("cp") or inf.startswith("res"): continue
		with open(join(rdir, inf), encoding="utf-8") as fin:
			lines = fin.readlines()
		if bl: fout = open(join(rdir, inf), "w", encoding="utf-8")
		else: fout = None

		fin, line = lines[1:], lines[0]
		if fout: fout.write(line)
		fields = line.strip("\r\n").split("\t")
		for line in fin:
			item = {}
			for k, v in zip(fields, line.strip("\r\n").split("\t")):
				if k == "time":
					if 0 < float(v) < 100: v = 100
					v = round(math.log(1+float(v), 10))
				elif k not in mfa: pass
				elif mfa[k][2] == "perc":
					v = v.rsplit("=", 1)[0]
					a, b = v.split("/")
					v = float(a)/float(b)*100
				elif mfa[k][2] == "int":
					v = int(v)
				elif mfa[k][2] == "float":
					v = float(v) if v.strip() else 0
				item[k] = v
			if item["method"] in bl: continue
			key = tuple(str(item.pop(f)) for f in spec_fields)
			agg = item.pop(reduce_field)
			results[key][agg].append(item)
			mtc_fields |= set(item.keys())
			if fout: fout.write(line)
		if fout: fout.close()
	puzset = set.intersection(*(set(v.keys()) for v in results.values()))
	print("Aggregate the results of", len(puzset), "common puzzles")
	mtc_fields = sorted(list(mtc_fields), key=lambda x:mfa[x])
	print("Metric fields", mtc_fields)
	with open(ouf, "w", encoding="utf-8") as fout:
		fout.write("%s\t%s\tRepeats\n"%("\t".join(spec_fields), "\t".join(mtc_fields)))
		for key, res in sorted(results.items()):
			r = defaultdict(float)
			repeats = 9e9
			for agg, items in res.items():
				if agg in puzset and len(items) < repeats: 
					repeats = len(items)
			for agg, items in res.items():
				if agg in puzset:
					for item in items[:repeats]:
						for k, v in item.items():
							r[k] += v/repeats
			for k in r: r[k] /= len(puzset)
			fout.write("%s\t%s\t%d\n"%("\t".join(key), "\t".join(mfa[k][1]%r[k] for k in mtc_fields), repeats))
if __name__ == "__main__":
	aggregate_results(infs="outputs/overall" if len(sys.argv) < 2 else sys.argv[1], bl=set() if len(sys.argv) < 3 else set(sys.argv[2].split(",")))