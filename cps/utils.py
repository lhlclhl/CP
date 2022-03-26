import argparse, traceback, json, os, unidecode, re, sys

def save_list(st, ofn):
	with open(ofn, "w", encoding = "utf-8") as fout:
		for k in st:
			if type(k) == tuple or type(k) == list:
				fout.write("%s\n"%"\t".join(k))
			elif type(k) == dict:
				fout.write("%s\n"%json.dumps(k, ensure_ascii=False))
			else:
				fout.write("%s\n"%str(k))
def load_list(listfile):
	return [l if l=="\n"else l.strip("\n") for l in open(listfile, encoding="utf-8")]
def csv_iter(dfile, delim='\t', encoding='utf-8'):
	try:
		if dfile.endswith(".bz2"): 
			import bz2file
			fp = bz2file.open(dfile, "rb")
		elif dfile.endswith(".gz"): 
			import gzip
			fp = gzip.GzipFile(dfile, "rb")
		elif dfile.endswith(".zip"):
			from zipfile import ZipFile
			zf = ZipFile(dfile)
			fp = zf.open(os.path.split(dfile)[-1].replace(".zip", ".txt"), "r")
		else: fp = open(dfile, 'rb')
	except Exception: traceback.print_exc()
	with fp as fin:
		if delim: 
			for line in fin: yield line.decode(encoding, errors="ignore").rstrip('\n\r').split(delim)
		else: 
			for line in fin: yield line.decode(encoding, errors="ignore")
def parse_args(params, args=None):
	parser = argparse.ArgumentParser()
	for k, v in params.items():
		t = str if v is None else type(v)
		if t == bool:
			parser.add_argument('--%s'%k, default=v, action='store_true')
		else:
			parser.add_argument('--%s'%k,  type=t, default=v)
	pargs, unkargs = parser.parse_known_args(args=args)
	return pargs, unkargs
def get_banner(args, params, blacklist):
	lparams = []
	for k, v in params.items():
		if k in blacklist: continue
		if hasattr(args, k) and getattr(args, k) != v: lparams.append((k, getattr(args, k)))
	if not lparams: return "default"
	else: return ",".join("%s=%s"%(k, v)for k, v in lparams)
def pprior(keys, key_priors={}):
	return (-key_priors.get(keys[0], 0),)+keys

class BaseObject:
	'''
	Base object with parameter management
	'''
	@property
	def params(self): return {
	}
	@property
	def options(self): return {
	}
	@property
	def banner(self): return self._banner()
	def _banner(self, black=set()): 
		display_fields = [(arg, val) for arg, val in sorted(self.params.items(),key=pprior)\
							if arg not in black and type(val) in {str, int, float}]
		model_banner = ",".join("%s=%s"%(arg,getattr(self, arg)) for arg, val in display_fields if getattr(self, arg)!=val)
		if not model_banner: model_banner = "default"
		return model_banner
	def __str__(self):
		return "%s,%s"%(type(self).__name__, self.banner)
	def __init__(self, args=sys.argv[1:], **kwargs):
		self._input_args = args
		if type(args) != argparse.Namespace: 
			parser = argparse.ArgumentParser(args)
			for k, v in dict(self.options, **self.params).items():
				t = str if v is None else type(v)
				if t == bool:
					parser.add_argument('--%s'%k, default=v, action='store_true')
				else:
					parser.add_argument('--%s'%k,  type=t, default=v)
			args, _ = parser.parse_known_args(args=args)
		
		if hasattr(args, "config"): 
			config = json.load(open(args.config, encoding="utf-8"))
		else: config = {}

		for k, v in dict(self.options, **self.params).items():
			tc = (lambda x:x) if v is None else type(v) # type class of param
			if k in kwargs:
				v = tc(kwargs[k])
			elif hasattr(args, k):
				v = tc(getattr(args, k))
			elif k in config:
				v = tc(config[k])
			setattr(self, k, v)

		for ifunc in self.initializations:
			ifunc(args, **kwargs)

		print(str(self))
	@property
	def initializations(self): return []
