import pytorch_lightning as pl
from tango.common import Registrable


class EventPredictor(Registrable, pl.LightningModule):
    def predict(self, batch):
        raise NotImplementedError
