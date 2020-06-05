# -*- coding: utf-8 -*-
import os
import re
def findFiles(path): return glob.glob(path)

def readLines(data_path):
	lines = open(data_path, encoding='utf-8').read().strip().split('\n')
	return lines

class Data:
	def __init__(self, name, lang=''):
		self.raw_data = os.path.join(lang, name+'.txt')
		self.BOI_data = os.path.join(lang, name+'_BOI.txt')

	def spliteKeyWord(self,str):
	    regex = r"[\u4e00-\ufaff]|[0-9]+|[a-zA-Z]+\'*[a-z]*"
	    matches = re.findall(regex, str, re.UNICODE)
	    return matches

	def split(self, string):
		return self.spliteKeyWord(string)
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
	parser.add_argument('-l', '--language', action='store', dest='language', default='cmn_CHN', help='language code')
	args = parser.parse_args()

	# data = Data(args.domain_name)
	# data.Raw_data()

	# exit()
	domains = os.listdir(args.language)
	for domain in domains:
		domain = domain.split('.')[0]
		print(domain)
		data = Data(domain, args.language)
		data.Raw_data()