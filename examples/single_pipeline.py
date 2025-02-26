import pandas as pd
import numpy as np
import sys
import utils

# Include Streamline
sys.path.append('src')
from streamline import Streamline, WorkloadPredictor, Pipeline, NeuralNetwork, PolynomialRegression


def main():
    # Get the pipeline
    pipeline_1 = utils.get_pipeline_1()

    # Get historical data of the pipeline, e.g. using the rest api of Apache Flink
    historical_data_pipeline_1 = pd.read_csv("examples/data/historical_data_pipeline_1.csv")

    # Measure workload of the sources of pipeline, e.g. using the rest api of Apache Flink
    observed_source_workloads_pipeline_1 = {"v_0": 50 + (np.sin(np.linspace(0, 10 * np.pi, 500)) + 1) * (2000 - 50) / 2}

    # Create Streamline object
    streamline = Streamline()

    # Prepare workload prediction (train the transformer model for each source operator)
    t_0 = streamline.prepare_workload_predictor(observed_source_workloads_pipeline_1["v_0"], WorkloadPredictor(epochs=200, output_size=100))

    # Prepare for configuration optimizer (train neural network and polynomial regression models for the pipeline)
    nn_pipeline_1, pr_pipeline_1 = streamline.prepare_configuration_optimizer(NeuralNetwork(historical_data_pipeline_1, epochs=200),
                                                                              PolynomialRegression(historical_data_pipeline_1, create_plots=True))

    # Predict the future workload of each operator
    predicted_workload_pipeline_1 = streamline.predict_workload(pipeline_1, historical_data_pipeline_1, observed_source_workloads_pipeline_1["v_0"], t_0)

    # (Optional) Plot future workload of the source operator for visibility
    t_0.plot(t_0.predict(observed_source_workloads_pipeline_1["v_0"][-t_0.output_size:]), 500, observed_source_workloads_pipeline_1["v_0"], "transformers_v_0")

    # Run the configuration optimizer to find configurations for each operator
    configurations_pipeline_1 = streamline.configuration_optimizer(pipeline_1, predicted_workload_pipeline_1, nn_pipeline_1, pr_pipeline_1, max_parallelism_to_check=5)

    # Schedule operators on resources
    placement_full, _, _, _ = streamline.schedule(configurations_pipeline_1, utils.get_resources())

    # (Optional) print placement
    print(placement_full)


if __name__ == "__main__":
    main()
