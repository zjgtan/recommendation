# coding:

from sklearn.utils import shuffle

class BatchReader(object):
    """reader
    """
    def __init__(self, samples, make_batch):
        self.samples = samples
        self.make_batch_records = make_batch

        self.batch_size = 128

    def next_batch(self, is_shuffle = False):
        """batch
        """
        samples = self.samples

        if is_shuffle:
            samples = shuffle(self.samples)
        
        
        count = 0

        records = {}
        for record in samples:
            if count == self.batch_size:
                yield self.make_batch_records(records)
                records = {}
                count = 0

            count += 1

            for key, value in record.iteritems():
                 records.setdefault(key, [])
                 records[key].append(value)

        if len(records) != 0:
            yield self.make_batch_records(records)


