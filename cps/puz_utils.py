import numpy as np, random, string, re, unidecode

try:
	from .puz_utils1 import *
	print("Cython utils imported")
except Exception:
	print("No Cython Module, defining python functions which will be less efficient")
	def concate_bytes(byts):
		return bytearray(byts).decode().replace("\x00", "")
	concate_bytes_array = concate_bytes
	def get_string(grid, x, y, direction, length):
		if direction == 1:
			return concate_bytes(grid[x,y:y+length])
		else:
			return concate_bytes(grid[x:x+length,y])
	def fill(grid, direction, x, y, word, new=True):
		if new: ngrid = grid.copy()
		else: ngrid = grid
		if direction == 1:
			ngrid[x,y:y+len(word)] = np.array(list(word), dtype="S")
		else:
			ngrid[x:x+len(word),y] = np.array(list(word), dtype="S")
		return ngrid
	def match(s1, s2):
		if len(s1) != len(s2): return False
		for c1, c2 in zip(s1, s2):
			if c1 != "*" and c2 != "*" and c1 != c2: return False
		return True

directions = ["down", "across"]
block_sign = b"."
blank_sign = b"*"
blank = "___"

re_ref = re.compile("([\d]+)-(down|across)")
re_short = re.compile("(abbr.|for short)")
re_e2g = re.compile("[^A-Z]+")
def string2fill(s):
	return re_e2g.sub("", unidecode.unidecode(s.upper())).strip()

def tokenize(text):
	tokens = []
	for t in text.split():
		t = t.strip('"')
		st = t.lower().strip().strip(string.punctuation)
		swb = t.startswith(blank)
		if swb: tokens.append(blank)
		if st: tokens.append(st)
		if not swb and t.endswith(blank): tokens.append(blank)
	return tokens

def grid2string(grid):
	return "\n".join(concate_bytes_array(row) for row in grid)

def evaluate(ogrid, tgrid, num2pos, answers):
	n_letters = n_words = 0
	for orow, trow in zip(ogrid, tgrid):
		for o, t in zip(orow, trow):
			if t != b"." and o == t: n_letters += 1
	for i in range(len(answers)):
		for k, v in answers[i].items():
			s = get_string(ogrid, *(num2pos[k]), i, len(v))
			if s == v: n_words += 1
	return n_words, n_letters

def construct_grid(puzzle):
	'''
	clues: [
		{index:(text, length)}, # down
		{...}, # across
	]
	'''
	shape = (puzzle["size"]["rows"], puzzle["size"]["cols"])
	agrid = []
	pos2num = np.zeros(shape, dtype="i2")
	num2pos = {}
	for r in range(shape[0]):
		agrid.append( puzzle["grid"][r*shape[1]:(r+1)*shape[1]] )
		for c in range(shape[1]):
			gn = puzzle["gridnums"][r*shape[1]+c]
			if gn: 
				pos2num[r,c] = gn
				num2pos[gn] = (r, c)
	agrid = np.array(agrid, dtype="S")
	grid = np.array([[x if x == b"." else b"*" for x in row] for row in agrid], dtype="S")

	clues = [{}, {}]
	answers = [{}, {}]
	for i, key in enumerate(directions):
		for clue, ans in zip(puzzle["clues"][key], puzzle["answers"][key]):
			num, text = clue.split(". ", 1)
			num = int(num)
			pos = list(num2pos[num])
			while pos[i] < shape[i] and agrid[tuple(pos)] != b".":
				pos[i] += 1
			clues[i][num] = (text, pos[i]-num2pos[num][i])
			answers[i][num] = ans

	num2pos_arr = np.zeros((max(num2pos.keys())+1, 2), dtype="i2")
	for i in range(len(num2pos_arr)): num2pos_arr[i] = num2pos.get(i, -1)
	num2pos = num2pos_arr

	# check answers in grid
	for i, direction in enumerate(directions):
		for k, v in answers[i].items():
			u = get_string(agrid, *num2pos[k], i, clues[i][k][1])
			assert v == u, "inconsistent %s ans: %s"%(direction, str((k, v, u)))
	return grid, num2pos, pos2num, clues, answers, agrid

def construct_from_html(html_content):
	from bs4 import BeautifulSoup
	soup = BeautifulSoup(html_content, "lxml")
	title = soup.title.text
	grid_area = soup.find("div", {"class": "grid-area"}).find("div", {"class":"crossword"})
	clue_area = soup.find("div", {"class": "clues-area"})
	aclues = clue_area.find("div", {"class": "aclues"})
	dclues = clue_area.find("div", {"class": "dclues"})

	grid = [[]]
	num2pos = {}
	for item in grid_area.children:
		if item.name != "div": continue
		if "endRow" in item.attrs["class"]: grid.append([])
		elif item.find("img", {"alt": "black cell"}):
			grid[-1].append(block_sign.decode())
		else:
			cluenum = item.find("span", {"class": "cluenum-in-box"})
			if cluenum:
				num2pos[int(cluenum.text)] = (len(grid)-1, len(grid[-1]))
			grid[-1].append(blank_sign.decode())
	if not grid[-1]: grid.pop()
	grid = np.array(grid, dtype="S")
	n_rows, n_cols = grid.shape

	clues = [{}, {}]
	for i, clue_block in enumerate([dclues, aclues]):
		for clue_div in clue_block.find_all("div", {"class": "clueDiv"}):
			cluenum = clue_div.find("div", {"class": "clueNum"})
			clue = clue_div.find("div", {"class": "clue"})

			for s in cluenum.select('div'): s.extract()
			num = int(cluenum.text)
			x, y = num2pos[num]
			if i == 0: # down
				for k in range(x+1, n_rows+1):
					if k < n_rows and grid[k][y] == block_sign: break
				length = k - x
			else: # across
				for k in range(y+1, n_cols+1):
					if k < n_rows and grid[x][k] == block_sign: break
				length = k - y
			clues[i][num] = (clue.text, length)

	pos2num = {pos: num for num, pos in num2pos.items()}
	answers = agrid = None
	return title, grid, num2pos, pos2num, clues, answers, agrid
def construct_from_text(content):
	''' layout:
		grids in 0/1
		[EMPTY LINE]
		across clues (maybe with no. as a new line)
		[EMPTY LINE]
		down clues (maybe with no. as a new line)
	'''
	lines = [line.strip() for line in content.split("\n")]
	# read grid
	grid = []
	for line in lines:
		if line.strip():
			grid.append(line.strip())
		else: break
	grid = np.array([list(row) for row in grid], dtype="S1")
	n_rows, n_cols = grid.shape
	offset = len(grid)

	# read across clues
	while not lines[offset].strip(): offset += 1
	aclues = []
	while lines[offset].strip():
		try: int(lines[offset].strip())
		except Exception: aclues.append(lines[offset].strip())
		offset += 1
	
	# read down clues
	while not lines[offset].strip(): offset += 1
	dclues = []
	while offset < len(lines) and lines[offset].strip():
		try: int(lines[offset].strip())
		except Exception: dclues.append(lines[offset].strip())
		offset += 1
	
	num2pos, pos2num = {}, {}
	num = 0
	clue_no_across, clue_no_down = [], []
	for i in range(n_rows):
		for j in range(n_cols):
			if grid[i][j] == b"0":
				acs = j == 0 or grid[i][j-1] == b"1"
				dns = i == 0 or grid[i-1][j] == b"1"
				if acs or dns:
					num += 1
					num2pos[num] = (i, j)
					pos2num[i, j] = num
					if acs: 
						for k in range(j+1, n_cols+1):
							if k < n_cols and grid[i][k] == b"1": break
						clue_no_across.append((num, k-j))
					if dns: 
						for k in range(i+1, n_rows+1):
							if k < n_rows and grid[k][j] == b"1": break
						clue_no_down.append((num, k-i))
	assert len(clue_no_across) == len(aclues), "across no. (%d) inconsistent with across clues (%d)" % (len(clue_no_across), len(aclues))
	assert len(clue_no_down) == len(dclues), "down no. (%d) inconsistent with down clues (%d)" % (len(clue_no_down), len(dclues))
	print("number of across clues", len(aclues))
	print("number of down clues", len(dclues))
	clues = [
		{num: (clue, l) for (num, l), clue in zip(clue_no_down, dclues)},
		{num: (clue, l) for (num, l), clue in zip(clue_no_across, aclues)}
	]
	for i in range(n_rows):
		for j in range(n_cols):
			if grid[i][j] == b"0": grid[i][j] = blank_sign
			if grid[i][j] == b"1": grid[i][j] = block_sign
	answers = agrid = None
	return None, grid, num2pos, pos2num, clues, answers, agrid

