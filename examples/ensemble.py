import pandas as pd
import numpy as np
import sys
import utils

# Include Streamline
sys.path.append('src')
from streamline import Streamline, WorkloadPredictor, Pipeline, NeuralNetwork, PolynomialRegression


def main():
    # Get all pipelines within the ensemble
    pipeline_1 = utils.get_pipeline_1()
    pipeline_2 = utils.get_pipeline_2()

    # Get historical data of the pipelines, e.g. using the rest api of Apache Flink
    historical_data_pipeline_1 = pd.read_csv("examples/data/historical_data_pipeline_1.csv")
    historical_data_pipeline_2 = pd.read_csv("examples/data/historical_data_pipeline_2.csv")

    # Measure workload of the sources of pipelines, e.g. using the rest api of Apache Flink
    observed_source_workloads_pipeline_1 = {"v_0": 50 + (np.sin(np.linspace(0, 10 * np.pi, 500)) + 1) * (2000 - 50) / 2}
    observed_source_workloads_pipeline_2 = {"v_5": 50 + (np.cos(np.linspace(0, 10 * np.pi, 500)) + 1) * (2000 - 50) / 2}

    # Create Streamline object
    streamline = Streamline()

    # Prepare workload prediction (train the transformer model for each source operator)
    t_0 = streamline.prepare_workload_predictor(observed_source_workloads_pipeline_1["v_0"], WorkloadPredictor(epochs=200, output_size=100))
    t_5 = streamline.prepare_workload_predictor(observed_source_workloads_pipeline_2["v_5"], WorkloadPredictor(epochs=200, output_size=100))

    # Prepare for configuration optimizer (train neural network and polynomial regression models for each pipeline)
    nn_pipeline_1, pr_pipeline_1 = streamline.prepare_configuration_optimizer(NeuralNetwork(historical_data_pipeline_1, epochs=200),
                                                                              PolynomialRegression(historical_data_pipeline_1, create_plots=True))
    nn_pipeline_2, pr_pipeline_2 = streamline.prepare_configuration_optimizer(NeuralNetwork(historical_data_pipeline_2, epochs=200),
                                                                              PolynomialRegression(historical_data_pipeline_2, create_plots=True))

    # Predict the future workload of each operator
    predicted_workload_pipeline_1 = streamline.predict_workload(pipeline_1, historical_data_pipeline_1, observed_source_workloads_pipeline_1["v_0"], t_0)
    predicted_workload_pipeline_2 = streamline.predict_workload(pipeline_2, historical_data_pipeline_2, observed_source_workloads_pipeline_2["v_5"], t_5)

    # (Optional) Plot future workload of source operators for visibility
    t_0.plot(t_0.predict(observed_source_workloads_pipeline_1["v_0"][-t_0.output_size:]), 500, observed_source_workloads_pipeline_1["v_0"], "transformers_v_0")
    t_5.plot(t_5.predict(observed_source_workloads_pipeline_2["v_5"][-t_5.output_size:]), 500, observed_source_workloads_pipeline_2["v_5"], "transformers_v_5")

    # Run the configuration optimizer to find configurations for each operator
    configurations_pipeline_1 = streamline.configuration_optimizer(pipeline_1, predicted_workload_pipeline_1, nn_pipeline_1, pr_pipeline_1, max_parallelism_to_check=5)
    configurations_pipeline_2 = streamline.configuration_optimizer(pipeline_2, predicted_workload_pipeline_2, nn_pipeline_2, pr_pipeline_2, max_parallelism_to_check=5)

    # Merge the configurations of all operators
    all_configurations = {**configurations_pipeline_1, **configurations_pipeline_2}

    # Schedule operators on resources
    placement_full, _, _, _ = streamline.schedule(all_configurations, utils.get_resources())

    # (Optional) print placement
    print(placement_full)


if __name__ == "__main__":
    main()
