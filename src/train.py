# coding: utf8

import sys
from dataset import ML1M
from torch import nn
import torch

from tensorboardX import SummaryWriter


ml1m_dataset = ML1M("../ml-1m/")
print >> sys.stderr, "load dataset..."
ml1m_dataset.load_dataset()

from deepfm import DeepFM

model = DeepFM(6040, 3953)

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters())

writer = SummaryWriter("log")
niter = 0

for epoch in range(10):
    for batch_id, batch_records in enumerate(ml1m_dataset.next_batch(128)):
        preds = model(batch_records)
        scores = batch_records["score"]

        loss = criterion(preds, scores)
        if batch_id % 100 == 0:
            print "iter: %d, batch_id: %d, loss: %f" % (
                    epoch, batch_id, loss.item())
            #writer.add_scalar("Train/Loss", loss.item(), niter)
            #niter += 1
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    metrics = evaluate(model)
