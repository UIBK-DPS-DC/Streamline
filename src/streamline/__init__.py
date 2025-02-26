import logging

# Configure logging to log messages to both a file and the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # Log to a file
        logging.StreamHandler()  # Log to console
    ]
)

# Import required components
from .workload_predictor import WorkloadPredictor
from .configuration_selector import ConfigurationSelector
from .scheduler import Scheduler
from .pipeline import Pipeline
from .neural_network import NeuralNetwork
from .polynomial_regression import PolynomialRegression
from .controller import Streamline

# Define the publicly available objects when importing from this module
__all__ = ["WorkloadPredictor", "ConfigurationSelector", "Scheduler", "Pipeline", "Streamline"]
