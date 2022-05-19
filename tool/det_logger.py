import os
import logging
import time
from typing import Dict, List
import json


class DetLogger:
    def __init__(self, workspace: str, level: str):
        if not os.path.isdir(workspace):
            os.mkdir(workspace)
        self._workspace: str = workspace

        self._level: int = logging.INFO if level == "INFO" else logging.DEBUG
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
        self._logger = logging.getLogger("message")
        self._logger.setLevel(self._level)

        fileHandler = logging.FileHandler(os.path.join(workspace, "ouput.log"))
        fileHandler.setFormatter(formatter)
        fileHandler.setLevel(self._level)
        self._logger.addHandler(fileHandler)

        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        streamHandler.setLevel(self._level)
        self._logger.addHandler(streamHandler)
        self._time: float = time.time()

        self._save_path: str = os.path.join(workspace, "metric.txt")

    def reportTime(self, name: str):
        current: float = time.time()
        self._write(name + " - time: {}".format(current - self._time))

    def reportMetric(self, name: str, metric: Dict):
        self.reportDelimitter()
        self.reportTime(name)
        keys: List = list(metric.keys())
        for key in keys:
            self._write("\t- {}: {}".format(key, metric[key]))
        self.reportDelimitter()
        self.reportNewLine()

    def writeFile(self, metric: Dict):
        with open(self._save_path, 'a', encoding='utf=8') as f:
            f.write(json.dumps(metric))
            f.write("\n")

    def reportDelimitter(self):
        self._write("-" * 33)

    def reportNewLine(self):
        self._write("")

    def _write(self, message: str):
        if self._level == logging.INFO:
            self._logger.info(message)
            return
        self._logger.debug(message)
