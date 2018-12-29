# coding: utf8

import sys
sys.path.append("../dataset")
sys.path.append("../evaluation")

from logzero import logger

from ml1m_rate_dataset import ML1M
from fm_model import FMModel
from cross_validation import CrossValidation

logger.info("load dataset...")
dataset = ML1M("../../ml-1m/")
logger.info("end...")

model = FMModel(6040, 3953)
cross_validator = CrossValidation(dataset)
cross_validator.do_k_fold_validation(model, 10)
