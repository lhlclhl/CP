#!python
#cython: language_level=3
#cmd: cythonize -a -i cps/puz_utils1.pyx
cpdef bint match(str s1, str s2):
	if len(s1) != len(s2): return False
	for i in range(len(s1)):
		if s1[i] != "*" and s2[i] != "*" and s1[i] != s2[i]: return False
	return True

cpdef str concate_bytes(char[:, :] byts):
	return bytearray(byts).decode().replace("\x00", "")

cpdef str concate_bytes_array(char[:] byts):
	return bytearray(byts).decode().replace("\x00", "")

cpdef str get_string(char[:, :] grid, int x, int y, int direction, int length):
	if direction == 1:
		return concate_bytes_array(grid[x,y:y+length])
	else:
		return concate_bytes_array(grid[x:x+length,y])

cpdef char[:,:] fill(char[:,:] grid, int direction, int x, int y, str word, bint new=True):
	cdef char[:,:] ngrid
	if new: ngrid = grid.copy()
	else: ngrid = grid
	cdef int k
	if direction == 1:
		for k in range(len(word)):
			ngrid[x,y+k] = word[k]
	else:
		for k in range(len(word)):
			ngrid[x+k,y] = word[k]
	return ngrid

'''
est=21.51/76708=2.804e-04
act=71.53/61748=1.158e-03
		clueG=46.93/2288190=2.051e-05
		vocabG=2.47/1073325=2.300e-06

# cythonize match()
est=28.01/99675=2.810e-04
act=62.48/74626=8.373e-04
	clueG=32.34/2679229=1.207e-05
	vocabG=3.02/1259457=2.400e-06

# cythonize get_string() and concat_bytes()
est=14.74/110305=1.336e-04
act=74.37/84791=8.771e-04
		clueG=39.23/3041844=1.290e-05
		vocabG=3.36/1443783=2.330e-06

# cythonize fill()
est=14.52/129649=1.120e-04
act=69.13/99756=6.930e-04
		clueG=44.29/3600783=1.230e-05
		vocabG=3.58/1695778=2.110e-06

# lazy strategy for fill()
est=15.26/144011=1.059e-04
act=69.76/110713=6.301e-04
		clueG=47.59/4002577=1.189e-05
		vocabG=3.88/1891753=2.052e-06
est=15.84/143708=1.102e-04
act=70.59/110747=6.374e-04
		clueG=49.12/4013402=1.224e-05
		vocabG=3.92/1897886=2.067e-06
		sort=12.67/4013402=3.156e-06

# make a list for candidate iteration
est=17.01/152229=1.117e-04
act=68.44/114066=6.000e-04
		clueG=47.06/4030796=1.168e-05
		vocabG=3.81/1909634=1.994e-06
		sort=12.57/4030796=3.118e-06
est=17.66/158918=1.112e-04
act=67.18/114867=5.848e-04
		clueG=46.16/4042035=1.142e-05
		vocabG=3.82/1978430=1.932e-06
		sort=12.22/4042035=3.024e-06

# optimize match
est=22.86/211678=1.080e-04
act=57.55/163740=3.514e-04
		clueG=26.78/5912944=4.529e-06
		vocabG=5.20/2804327=1.854e-06
		sort=18.31/5912944=3.096e-06
est=23.84/220078=1.083e-04
act=55.28/162408=3.404e-04
		clueG=25.98/5781635=4.494e-06
		vocabG=4.91/2774400=1.769e-06
		sort=17.56/5781635=3.037e-06

# optimize sorting
est=29.41/270582=1.087e-04
act=44.54/196313=2.269e-04
		clueG=32.71/6847898=4.776e-06
		vocabG=5.68/3289310=1.727e-06
		sort=0.00/0=0.000e+00
est=30.97/285114=1.086e-04
act=42.04/195617=2.149e-04
		clueG=30.50/6748147=4.519e-06
		vocabG=5.46/3290348=1.661e-06
		sort=0.00/0=0.000e+00

# reduce estimation 
est=24.70/223557=1.105e-04
act=45.87/223557=2.052e-04
		clueG=32.69/7553946=4.327e-06
		vocabG=6.48/3853359=1.681e-06
		sort=0.00/0=0.000e+00
est=23.71/215066=1.102e-04
act=47.33/215066=2.201e-04
		clueG=34.64/7548143=4.589e-06
		vocabG=6.06/3701261=1.638e-06
		sort=0.00/0=0.000e+00

# buf clueG
est=31.78/299809=1.060e-04
act=18.16/299809=6.057e-05
        clueG=3.76/9807552=3.831e-07
        vocabG=5.14/3947588=1.303e-06
        sort=0.00/0=0.000e+00
584778 states, 299809 unique states
est=32.47/310093=1.047e-04
act=18.41/310093=5.936e-05
        clueG=3.93/10295049=3.816e-07
        vocabG=5.03/4095612=1.228e-06
        sort=0.00/0=0.000e+00
576442 states, 310093 unique states
'''