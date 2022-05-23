import os.path

import yaml
import torch
from dataset import DetLoader
from tool import DetLogger, DetAverager, DetCheckpoint
from typing import Dict, Tuple
import torch.optim as optim
import argparse
import warnings
from loss_model import LossModel


class DetTrainer:
    def __init__(self,
                 lossModel: Dict,
                 train: Dict,
                 valid: Dict,
                 optimizer: Dict,
                 checkpoint: Dict,
                 logger: Dict,
                 totalEpoch: int,
                 startEpoch: int,
                 lr: float,
                 factor: float,
                 **kwargs):
        self._device = torch.device('cpu')
        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        self._model: LossModel = LossModel(**lossModel, device=self._device)
        self._train = DetLoader(**train).build()
        self._valid = DetLoader(**valid).build()
        self._checkpoint: DetCheckpoint = DetCheckpoint(**checkpoint)
        self._logger: DetLogger = DetLogger(**logger)
        optimCls = getattr(optim, optimizer['name'])
        self._lr: float = lr
        self._factor: float = factor
        self._step = 0
        self._optim: optim.Optimizer = optimCls(**optimizer['args'],
                                                lr=self._lr,
                                                params=self._model.parameters())
        self._totalEpoch: int = totalEpoch + 1
        self._startEpoch: int = startEpoch
        self._curLR: float = lr
        self.totalLoss: DetAverager = DetAverager()
        self.probLoss: DetAverager = DetAverager()
        self.threshLoss: DetAverager = DetAverager()
        self.binaryLoss: DetAverager = DetAverager()

    def _loadCheckpoint(self):
        stateDict: Tuple = self._checkpoint.load(self._device)
        if stateDict is not None:
            self._model.load_state_dict(stateDict[0])
            self._optim.load_state_dict(stateDict[1])
            self._startEpoch = stateDict[2] + 1
            self._step = stateDict[3] + 1

    def _updateLR(self, epoch: int):
        rate: float = (1. - epoch / self._totalEpoch) ** self._factor
        self._curLR: float = rate * self._lr
        for groups in self._optim.param_groups:
            groups['lr'] = self._curLR

    def train(self):
        self._loadCheckpoint()
        self._logger.reportDelimitter()
        self._logger.reportTime("Start")
        self._logger.reportDelimitter()
        self._logger.reportNewLine()
        for i in range(self._startEpoch, self._totalEpoch):
            self._logger.reportDelimitter()
            self._logger.reportTime("Epoch {}".format(i))
            self._train_step(i)
        self._logger.reportDelimitter()
        self._logger.reportTime("Finish")
        self._logger.reportDelimitter()

    def _train_step(self, epoch: int):
        self._model.train()
        for batch in self._train:
            self._optim.zero_grad()
            batchSize: int = batch['img'].size(0)
            pred, loss, metric = self._model(batch)
            loss = loss.mean()
            loss.backward()
            self._optim.step()
            self.totalLoss.update(loss.item() * batchSize, batchSize)
            self.threshLoss.update(metric['threshLoss'].item() * batchSize, batchSize)
            self.binaryLoss.update(metric['binaryLoss'].item() * batchSize, batchSize)
            self.probLoss.update(metric['probLoss'].item() * batchSize, batchSize)
            self._step += 1
            if self._step % 100 == 0:
                validRS = self._valid_step()
                self._model.train()
                self._save({
                    'totalLoss': self.totalLoss.calc(),
                    'threshLoss': self.threshLoss.calc(),
                    'binaryLoss': self.binaryLoss.calc(),
                    'probLoss': self.probLoss.calc()
                }, validRS, epoch)
                self.totalLoss.reset()
                self.threshLoss.reset()
                self.binaryLoss.reset()
                self.probLoss.reset()

    def _valid_step(self):
        self._model.eval()
        totalLoss: DetAverager = DetAverager()
        probLoss: DetAverager = DetAverager()
        threshLoss: DetAverager = DetAverager()
        binaryLoss: DetAverager = DetAverager()
        with torch.no_grad():
            for batch in self._valid:
                batchSize: int = batch['img'].size(0)
                pred, loss, metric = self._model(batch)
                totalLoss.update(loss.mean().item() * batchSize, batchSize)
                threshLoss.update(metric['threshLoss'].item() * batchSize, batchSize)
                binaryLoss.update(metric['binaryLoss'].item() * batchSize, batchSize)
                probLoss.update(metric['probLoss'].item() * batchSize, batchSize)
        return {
            'totalLoss': totalLoss.calc(),
            'threshLoss': threshLoss.calc(),
            'binaryLoss': binaryLoss.calc(),
            'probLoss': probLoss.calc()
        }

    def _save(self, trainRS: Dict, validRS: Dict, epoch:int):
        self._logger.reportTime("Step {}".format(self._step))
        self._logger.reportMetric("Training", trainRS)
        self._logger.reportMetric("Validation", validRS)
        self._logger.writeFile({
            "training": trainRS,
            "validation": validRS
        })
        self._checkpoint.saveCheckpoint(self._step, epoch, self._model, self._optim)
        self._checkpoint.saveModel(self._model, self._step)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Training config")
    parser.add_argument("-p", '--path', type=str, help="path of config file")
    parser.add_argument("-d", '--data', default='', type=str, help="path of data")
    parser.add_argument("-i", '--imgType', default=0, type=int, help="type of image")
    parser.add_argument("-r", '--resume', default='', type=str, help="resume path")
    args = parser.parse_args()
    with open(args.path) as f:
        config: Dict = yaml.safe_load(f)
    if args.data.strip():
        for item in ["train", "valid"]:
            config[item]['dataset']['imgDir'] = os.path.join(args.data.strip(), item, "image/")
            config[item]['dataset']['tarFile'] = os.path.join(args.data.strip(), item, "target.json")
            config[item]['dataset']['imgType'] = args.imgType
    if args.resume.strip():
        config['checkpoint']['resume'] = args.resume.strip()
    trainer = DetTrainer(**config)
    trainer.train()
