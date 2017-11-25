import torch
from torch.autograd import Variable
import numpy as np

f = open('/home/nhamblet/proj/kbecomp-data/WN18RR/train.txt')
entIdx = dict()
relIdx = dict()
triples = []
for lineRaw in f:
	pieces = lineRaw.strip().split('\t')
	if pieces[0] not in entIdx:
		entIdx[pieces[0]] = len(entIdx)
	if pieces[2] not in entIdx:
		entIdx[pieces[2]] = len(entIdx)
	if pieces[1] not in relIdx:
		relIdx[pieces[1]] = len(relIdx)
	triples.append([entIdx[pieces[0]], relIdx[pieces[1]], entIdx[pieces[2]]])

f.close()

triples = np.array(triples)
h = torch.LongTensor(triples[:,0])
r = torch.LongTensor(triples[:,1])
t = torch.LongTensor(triples[:,2])

tn = np.random.randint(len(entIdx), size=len(triples))
tNeg = torch.LongTensor(tn)

D = 50

ent = torch.nn.Embedding(len(entIdx), D)
rel = torch.nn.Embedding(len(relIdx), D)

hpr = ent(h) + rel(r)
tt = ent(Variable(t))
ft = ent(Variable(tNeg)) # if this isn't a Variable, torch vomits, not entirely sure why
pos = (hpr - tt).norm(dim=1)
neg = (hpr - ft).norm(dim=1)
marginLoss = torch.nn.MarginRankingLoss(margin=1)
print marginLoss(pos, neg, Variable(torch.ones(len(triples))))

