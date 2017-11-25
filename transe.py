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
print len(triples), 'training triples'

vf = open('/home/nhamblet/proj/kbecomp-data/WN18RR/valid.txt')
valids = []
for lineRaw in vf:
	pieces = lineRaw.strip().split('\t')
	if pieces[0] in entIdx and pieces[1] in relIdx and pieces[2] in entIdx:
		valids.append([entIdx[pieces[0]], relIdx[pieces[1]], entIdx[pieces[2]]])

vf.close()
print len(valids), 'validation triples'

triples = np.array(triples)
h = Variable(torch.LongTensor(triples[:,0]), requires_grad=False)
r = Variable(torch.LongTensor(triples[:,1]), requires_grad=False)
t = Variable(torch.LongTensor(triples[:,2]), requires_grad=False)

tn = np.random.randint(len(entIdx), size=len(triples))
tNeg = Variable(torch.LongTensor(tn), requires_grad=False)

D = 50

ent = torch.nn.Embedding(len(entIdx), D)
rel = torch.nn.Embedding(len(relIdx), D)

e_lr = 1e-2
r_lr = 1e-4

e_optimizer = torch.optim.Adam(ent.parameters(), e_lr)
r_optimizer = torch.optim.Adam(rel.parameters(), r_lr)

e_optimizer.zero_grad()
r_optimizer.zero_grad()

valids = np.array(valids)
vh = torch.LongTensor(valids[:,0])
vr = torch.LongTensor(valids[:,1])
vt = torch.LongTensor(valids[:,2])

x = 0
for epoch in range(100):
	hpr = ent(h) + rel(r)
	tt = ent(t)
	ft = ent(tNeg)
	pos = (hpr - tt).norm(dim=1)
	neg = (hpr - ft).norm(dim=1)
	marginLoss = torch.nn.MarginRankingLoss(margin=1)
	loss = marginLoss(pos, neg, Variable(torch.ones(len(triples))))
	print loss.data[0]
        e_optimizer.zero_grad()
        r_optimizer.zero_grad()
	loss.backward()
        e_optimizer.step()
	r_optimizer.step()
 
	ent.train(False)
	rel.train(False)
        e_optimizer.zero_grad()
        r_optimizer.zero_grad()
        ent.zero_grad()
        rel.zero_grad()
	y = (torch.index_select(ent.weight.data, 0, vh) + torch.index_select(rel.weight.data, 0, vr) - torch.index_select(ent.weight.data, 0, vt)).norm(2, dim=1).sum()
        print y, y-x 
	x = y
        ent.zero_grad()
	rel.zero_grad()
	e_optimizer.zero_grad()
	r_optimizer.zero_grad()
	ent.train(True)
	rel.train(True)
