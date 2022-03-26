from .puz_utils import string2fill

def convert_wikititle(title):
	if title.endswith(")"): title = title[:title.find("(")]
	return string2fill(title)

class WikiMapper:
	def __init__(self, path, convert_func=lambda x:x, max_length=25):
		cnt = tot = 0
		self.mapper = {}
		with open(path, encoding="utf-8") as fin:
			for line in fin:
				lln = line.strip("\r\n").split("\t")
				qid = lln[0]
				titles = []
				for item in lln[1:]:
					wikiid, remain = item.split(",", 1)
					title, redirect = remain.rsplit(",", 1)
					ctitle = convert_func(title)
					if len(ctitle) <= max_length and ctitle not in titles:
						titles.append(ctitle)
						cnt += 1
					tot += 1
				self.mapper[qid] = titles
		self.id_to_titles = lambda x: self.mapper.get(x, [])
		print("Wikititles loaded: %d/%d"%(cnt, tot))

memory_mapper = None
def get_mapper():
	global memory_mapper
	if memory_mapper is None: 
		memory_mapper = WikiMapper("data/wiki/wikititles.txt", convert_func=convert_wikititle)
	return memory_mapper