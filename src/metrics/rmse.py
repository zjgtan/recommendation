# coding: utf8

import math

class RMSE(object):
    def eval(self, model, batch_reader):
        scores = []
        preds = []

        for batch_records in batch_reader.next_batch():
            output = model.net(batch_records)
            pred = output.data.numpy().tolist()[0]
            score = batch_records["score"].numpy().tolist()

            scores.extend(score)
            preds.extend(pred)

        fenzi = sum([(preds[i] - scores[i]) ** 2 for i in range(len(preds))])
        fenmu = len(preds)
        score = math.sqrt(fenzi / fenmu)

        return score


