# File: models/__init__.py

from models.default import MockModel
from models.transformer import Transformer

AvailableModels = {
    "MockModel": MockModel, 
    "Transformer" : Transformer,
}