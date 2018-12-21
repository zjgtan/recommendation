# coding: utf8

import sys
sys.path.append("../common/")
from dataset import ML1MExplicit
from torch import nn
import torch
import numpy as np
import math

from tensorboardX import SummaryWriter


ml1m_dataset = ML1MExplicit("../ml-1m/")
print >> sys.stderr, "load dataset..."
ml1m_dataset.load_dataset()
print >> sys.stderr, "end..."

from deepfm import DeepFM

def compute_accuracy_score(preds, scores):
    fenmu = sum(preds)
    fenzi = sum([1 for i in range(len(preds)) if preds[i] == 1 and scores[i] == 1])
    return fenzi * 1. / fenmu

def compute_recall_score(preds, scores):
    fenmu = sum(scores)
    fenzi = sum([1 for i in range(len(preds)) if preds[i] == 1 and scores[i] == 1])
    return fenzi * 1. / fenmu

def evaluate(model, iter_batch):
    scores = []
    preds = []

    for batch_records in iter_batch:
        output = model(batch_records)
        pred = output.data.numpy().tolist()[0]
        score = batch_records["score"].numpy().tolist()

        scores.extend(score)
        preds.extend(pred)

    fenzi = sum([(preds[i] - scores[i]) ** 2 for i in range(len(preds))])
    fenmu = len(preds)
    score = math.sqrt(fenzi / fenmu)

    return score

model = DeepFM(6040, 3953)
#criterion = nn.NLLLoss()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

writer = SummaryWriter("log")
niter = 0

batch_id = 0
for epoch in range(100):
    for batch_id, batch_records in enumerate(ml1m_dataset.next_batch(128)):
        #batch_records = ml1m_dataset.next_epoch()
        preds = model(batch_records)
        scores = batch_records["score"]

        loss = criterion(preds, scores)
        if batch_id % 1000 == 0:
            print "iter: %d, batch_id: %d, loss: %f" % (
                    epoch, batch_id, loss.item())
        #writer.add_scalar("Train/Loss", loss.item(), niter)
        #niter += 1
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step()

    print "start evaluate..."
    score = evaluate(model, ml1m_dataset.next_batch(128, True))
    print score
    writer.add_scalar("Train/rmse", score, epoch)
    score = evaluate(model, ml1m_dataset.next_batch(128, False))
    print score
    writer.add_scalar("Test/rmse", score, epoch)

