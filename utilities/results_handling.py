import numpy as np
import json
import os
import logging


logger = logging.getLogger(__name__)

class ResultHandler:
    def __init__(self, c=None,labels=None, binary_outputs=None, scalar_outputs=None, times=None, auroc=None, auprc=None, auroc_classes=None, auprc_classes=None, f_measure=None, f_measure_classes=None, challenge_metric=None, leads=[], fold=None, experiment=None, network="", accuracy=-1.0) -> None:
        self.c = c.tolist()
        self.labels = labels.tolist()
        self.binary_outputs=binary_outputs
        self.scalar_outputs=scalar_outputs.tolist()
        self.times=times.tolist()
        self.auroc=auroc
        self.auprc=auprc
        self.auroc_classes=auroc_classes.tolist()
        self.auprc_classes=auprc_classes.tolist()
        self.f_measure=f_measure
        self.f_measure_classes=f_measure_classes.tolist()
        self.challenge_metric=challenge_metric
        self.leads=list(leads)
        self.fold=fold
        self.experiment=experiment
        self.network=network
        self.accuracy=accuracy


    def save_json(self, filename):
        logger.debug(self.__dict__)
        as_json=json.dumps(self.__dict__)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as outfile:
            outfile.write(as_json)

