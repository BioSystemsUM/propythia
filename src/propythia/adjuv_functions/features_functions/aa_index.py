"""
 This module is used for obtaining the properties of amino acids or their pairs

from the aaindex database. You can freely use and distribute it. If you hava

any problem, you could contact with us timely!

Authors: Dongsheng Cao and Yizeng Liang.

Date: 2012.09.10

Email: oriental-cds@163.com
Altered and converted to python 3.6 by Ana Marta Sequeira 05/2019
"""
import sys, os

AALetter=["A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
_aaindex = dict()

#####################################################################################################


class Record:
	"""
	Amino acid index (AAindex) Record
	"""
	aakeys = 'ARNDCQEGHILKMFPSTWYV'
	def __init__(self):
		self.key = None
		self.desc = ''
		self.ref = ''
		self.authors = ''
		self.title = ''
		self.journal = ''
		self.correlated = dict()
		self.index = dict()
		self.comment = ''

	def extend(self, row):
		i = len(self.index)
		for x in row:
			self.index[self.aakeys[i]] = x
			i += 1

	def get(self, aai, aaj=None, d=None):
		assert aaj is None
		return self.index.get(aai, d)

	def __getitem__(self, aai):
		return self.get(aai)

	def median(self):
		x = sorted([_f for _f in list(self.index.values()) if _f])
		if len(x) % 2 == 1:
			return x[len(x)//2]
		return (sum(x[len(x)//2-1:len(x)//2+1]))/2

	def __str__(self):
		desc = self.desc.replace('\n', ' ').strip()
		return '%s(%s: %s)' % (self.__class__.__name__, self.key, desc)
		
#####################################################################################################


class MatrixRecord(Record):
	"""
	Matrix record for mutation matrices or pair-wise contact potentials
	"""
	def __init__(self):
		Record.__init__(self)
		self.index = []
		self.rows = dict()
		self.cols = dict()

	def extend(self, row):
		self.index.append(row)

	def _get(self, aai, aaj):
		i = self.rows[aai]
		j = self.cols[aaj]
		return self.index[i][j]

	def get(self, aai, aaj, d=None):
		try:
			return self._get(aai, aaj)
		except:
			pass
		try:
			return self._get(aaj, aai)
		except:
			return d

	def __getitem__(self, aaij):
		return self.get(aaij[0], aaij[1])

	def median(self):
		x = []
		for y in self.index:
			x.extend([_f for _f in y if _f])
		x.sort()
		if len(x) % 2 == 1:
			return x[len(x)//2]
		return sum(x[len(x)/2-1:len(x)/2+1])/2.0
#####################################################################################################


def search(pattern, searchtitle=True, casesensitive=False):
	"""
	Search for pattern in description and title (optional) of all records and
	return matched records as list. By default search case insensitive.

	:param pattern:
	:param searchtitle:
	:param casesensitive:
	:return:
	"""
	whatcase = lambda i: i
	if not casesensitive:
		pattern = pattern.lower()
		whatcase = lambda i: i.lower()
	matches = []
	for record in _aaindex.values():
		if pattern in whatcase(record.desc) or searchtitle and pattern in whatcase(record.title):
			matches.append(record)
	return matches

#####################################################################################################


def grep(pattern):
	"""
	Search for pattern in title and description of all records (case insensitive) and print results on standard output.

	:param pattern:
	:return:
	"""
	for record in search(pattern):
		print(record)
#####################################################################################################


def get(key):
	"""
	Get record for key

	:param key:
	:return:
	"""
	if len(_aaindex) == 0:
		init()
	return _aaindex[key]

#####################################################################################################


def _float_or_none(x):
	if x == 'NA' or x == '-':
		return None
	return float(x)

#####################################################################################################


def init(path=None, index='123'):
	"""
	Read in the aaindex files. You need to run this (once) before you can
	access any records. If the files are not within the current directory,
	you need to specify the correct directory path. By default all three
	aaindex files are read in.

	:param path:
	:param index:
	:return:
	"""
	index = str(index)
	if path is None:
		for path in [os.path.split(__file__)[0], '.']:
			if os.path.exists(os.path.join(path, 'aaindex' + index[0])):
				break
		print('path =', path, file=sys.stderr)
	if '1' in index:
		_parse(path + '/aaindex1', Record)
	if '2' in index:
		_parse(path + '/aaindex2', MatrixRecord)
	if '3' in index:
		_parse(path + '/aaindex3', MatrixRecord)
#####################################################################################################


def init_from_file(filename, type=Record):
	_parse(filename, type)
#####################################################################################################


def _parse(filename, rec, quiet=True):
	"""
	Parse aaindex input file. `rec` must be `Record` for aaindex1 and `MarixRecord` for aaindex2 and aaindex3.

	:param filename:
	:param rec:
	:param quiet:
	:return:
	"""
	if not os.path.exists(filename):
		import urllib.request, urllib.parse, urllib.error
		url = 'ftp://ftp.genome.jp/pub/db/community/aaindex/' + os.path.split(filename)[1]
#		print 'Downloading "%s"' % (url)
		filename = urllib.request.urlretrieve(url, filename)[0]
#		print 'Saved to "%s"' % (filename)
	f = open(filename)

	current = rec()
	lastkey = None
	for line in f:
		key = line[0:2]
		if key[0] == ' ':
			key = lastkey
		if key == '//':
			_aaindex[current.key] = current
			current = rec()
		elif key == 'H ':
			current.key = line[2:].strip()
		elif key == 'R ':
			current.ref += line[2:]
		elif key == 'D ':
			current.desc += line[2:]
		elif key == 'A ':
			current.authors += line[2:]
		elif key == 'T ':
			current.title += line[2:]
		elif key == 'J ':
			current.journal += line[2:]
		elif key == '* ':
			current.comment += line[2:]
		elif key == 'C ':
			a = line[2:].split()
			for i in range(0, len(a), 2):
				current.correlated[a[i]] = float(a[i+1])
		elif key == 'I ':
			a = line[1:].split()
			if a[0] != 'A/L':
				current.extend(list(map(_float_or_none, a)))
			elif list(Record.aakeys) != [i[0] for i in a] + [i[-1] for i in a]:
				print('Warning: wrong amino acid sequence for', current.key)
			else:
				try:
					assert list(Record.aakeys[:10]) == [i[0] for i in a]
					assert list(Record.aakeys[10:]) == [i[2] for i in a]
				except:
					print('Warning: wrong amino acid sequence for', current.key)
		elif key =='M ':
			a = line[2:].split()
			if a[0] == 'rows':
				if a[4] == 'rows':
					a.pop(4)
				assert a[3] == 'cols' and len(a) == 6
				i = 0
				for aa in a[2]:
					current.rows[aa] = i
					i += 1
				i = 0
				for aa in a[5]:
					current.cols[aa] = i
					i += 1
			else:
				current.extend(list(map(_float_or_none, a)))
		elif not quiet:
			print('Warning: line starts with "%s"' % (key))
		lastkey = key
	f.close()
#####################################################################################################


def get_aa_index1(name, path='.'):
	"""
	Get the amino acid property values from aaindex1

	:param name: name is the name of amino acid property (e.g., KRIW790103)
	:param path:
	:return: get_aa_index1(name). result is a dict form containing the properties of 20 amino acids
	"""
	init(path=path)
	name=str(name)
	temp=get(str.strip(name))
	res={}
	for i in AALetter:
		res[i]=temp.get(i)
	return res

#####################################################################################################


def get_aa_index23(name, path='.'):
	"""
	Get the amino acid property values from aaindex2 and aaindex3

	:param name: name is the name of amino acid property (e.g.,TANS760101,GRAR740104)
	:param path:
	:return: get_aa_index23(name). result is a dict form containing the properties of 400 amino acid pairs
	"""
	init(path=path)
	name=str(name)
	temp=get(str.strip(name))
	res={}
	for i in AALetter:
		for j in AALetter:
			res[i+j]=temp.get(i,j)
	return res
#####################################################################################################


if __name__ == "__main__":

	#	init(path='.')
	#	grep('volume')
	#	x = get('KRIW790103')
	#	print x
	#	print x.get('W')
	temp1=get_aa_index1('KRIW790103')
	print(len(temp1))
	
	temp2=get_aa_index23('TANS760101')
	print(len(temp2))
	temp2=get_aa_index23('GRAR740104')
	print(len(temp2))

