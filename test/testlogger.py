from tool.det_logger import DetLogger

logger = DetLogger(r'D:\workspace\project\db_pp\test\logger', 'DEBUG')
logger.reportDelimitter()
logger.reportTime("abc")
logger.reportDelimitter()
logger.reportNewLine()

metric = {
    "first": 1,
    "second": 2
}

logger.reportMetric("test", metric)
logger.writeFile(metric, logger.trainMetricPath)
