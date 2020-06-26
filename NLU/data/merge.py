import os

def check_dir(path):
    return os.path.exists(path)

def remove_old_file(path):
    if check_dir(path):
        os.remove(path)

def save_obj(obj, filename):
    remove_old_file(filename)
    json.dump(obj, open(filename, 'w', encoding="utf8"), ensure_ascii=False)

def findFiles(path): return glob.glob(path)

def readLines(data_path):
	lines = open(data_path, encoding='utf-8').read().strip().split('\n')
	return lines

def readTopKLines(data_path, line=1000):
	lines = []
	with open(data_path, encoding='utf-8') as f:
		for i, line in enumerate(f):
			if i == 1000:
				break
			lines.append(line.strip())
	return lines

def saveLines(lines, data_path):
	remove_old_file(data_path)
	with open(data_path, 'w', encoding ='utf-8') as f:
		for line in lines:
			f.write("{0}\n".format(line))


class Data:
	def __init__(self, name, lang=''):
		self.raw_data = os.path.join(lang, name+'.txt')
		self.BOI_data = os.path.join(lang, name+'_BOI.txt')

	def split(self, string):
		return list(string)
		# return " ".join(jieba.cut(word,HMM=True)).split(" ")


	def Raw_data(self):
		print("Loading data from Raw data...", flush = True)
		lines = readLines(self.raw_data)
		self.sents = []
		self.tags = []
		self.intents = []
		self.all_tags = []
		self.all_words = []
		self.all_intents = []

		# i = 0
		for line in lines:
			# i += 1
			# if i%1000 ==0:
			# 	print(i)
			segs = line.split('\t')
			intent = segs[0]
			# print(segs)
			# sent = segs[1]
			try:
				sent = segs[1]
			except:
				continue
			words = sent.split(' ')
			sent = []
			tag = []
			cur_tag = 'unk'  
			for word in words:
				if (word[0] != '<'):
					seg_list = self.split(word)
					num_seg = len(seg_list)
					sent = sent + seg_list
					if cur_tag == 'unk':
						tag = tag + ['O'] * num_seg
					elif first:
						tag = tag + ['B-'+cur_tag] +  ['I-' + cur_tag] * (num_seg-1)
						first = False
					else:
						tag = tag + ['I-' + cur_tag] * (num_seg)
				elif (word[0] == '<' and word[1] != '/'):
					cur_tag = word.replace('<','').replace('>','') 
					first = True  
				elif (word[0] == '<' and word[1] == '/'):
					cur_tag = 'unk'
			
			self.sents.append(sent)
			self.tags.append(tag)
			self.intents.append(intent)
			
			
			self.all_words = self.all_words + sent
			self.all_tags = self.all_tags + tag
			self.all_intents = self.intents

		print("Saving BOI data...", flush = True)
		with open(self.BOI_data, 'w', encoding ='utf-8') as f:
			for i in range(len(self.intents)):
				f.write("{0}".format(self.intents[i]) + '\t' +' '.join(self.sents[i]) + '\t' + ' '.join(self.tags[i]) + '\n')


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--domain', action='store', dest='domain_name', default='auto_only-nav-distance',help='domain name')
	parser.add_argument('-l', '--language', action='store', dest='language', default='cmn_CHN-word', help='language code')
	args = parser.parse_args()

	# data = Data(args.domain_name)
	# data.Raw_data()

	# exit()
	import glob
	path = os.path.join(args.language, "*BOI*")
	domains = glob.glob(path)
	# domains = os.listdir(path)
	lines = []
	lines_less = []
	for domain in domains:
		# domain = domain.split('\\')[-1].split('.')[0]
		lines_less += readTopKLines(domain)
		lines += readLines(domain)

	saveLines(lines,'data.txt')
	saveLines(lines_less,'data_less.txt')
		# print(domain)
		# data = Data(domain, args.language)
		# data.Raw_data()