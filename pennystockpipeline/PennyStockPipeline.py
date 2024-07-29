## #!/usr/bin/env python

## PennyStockPipeline
## This is a custom pipeline created for Penny Stock Prediction Project under CSE-115-UCSC (@Muktadir). 
## Please refer to the PennyStockPipeline.md for details of this pipeline

from .pipeline import PennyStockDataPipeline, PennyStockFeaturePipeline, PennyStockModelPipeline  #, PenyStockTrainingPipeline, PennyStockEvaluationPipeline
import csv, json, os

class PennyStockPipeline():

    def __init__(self, someArrayOfSubPipelines)
        pass