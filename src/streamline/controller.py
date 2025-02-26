import logging
from streamline.configuration_selector import ConfigurationSelector
from streamline.scheduler import Scheduler

# Create a logger instance
logger = logging.getLogger(__name__)


class Streamline:

    def prepare_workload_predictor(self, workload, transformer):

        # Train the transformer model
        logger.info(f"Starting Workload Predictor training using Transformers...")
        transformer.train_(workload)

        return transformer

    def predict_workload(self, pipeline, df, observed_source_workload, transformers):
        logger.info(f"Predicting workload of operators using Transformers and aspect ratio...")
        predicted_workload = {}
        ratio = {}

        # Iterate over all operators in a pipeline
        for operator in pipeline.list_vertices():
            operator_id = operator["id"]

            # Calculate the ratio of the operator
            filtered_df = df[df["Vertex Id"] == operator_id]
            mean_values = filtered_df.select_dtypes(include='number').mean()
            ratio[operator_id] = mean_values["TP_in"] / mean_values["TP_out"]

        # Use transformers to predict load of source operators
        source_load = transformers.predict(observed_source_workload[-transformers.output_size:])

        # Calculate load of remaining operators using the ratio
        predicted_workload = {**predicted_workload, **pipeline.get_load(source_load, ratio)}

        return predicted_workload

    def prepare_configuration_optimizer(self, neural_network, polynomial_regression):

        # Train the neural network
        logger.info(f"Starting Neural Network training...")
        neural_network.train_model()

        # Train the polynomial regression
        logger.info(f"Starting Polynomial Regression training...")
        return neural_network, polynomial_regression.generate_models_for_operators()

    def configuration_optimizer(self, pipeline, forecast_load, neural_network_model, polynomial_regression_models, max_parallelism_to_check):
        configurations = {}

        # Initial predecessor parallelism
        pred_parallelism = 1

        # Iterate over each operator in the pipeline
        for operator in pipeline.list_vertices():
            # Find configuration for operator
            configuration_selector = ConfigurationSelector(forecast_load[operator["id"]], operator["type"], neural_network_model, polynomial_regression_models, pred_parallelism)
            configuration_selector.solve(max_parallelism_to_check)

            # Retrieve the best configuration based on a weighted objective function
            best_configuration, best_score, best_obj = configuration_selector.get_best([0.5, 0.25, 0.25])

            # Store the best configuration for this operator
            configurations[operator["id"]] = {
                "parallelism": best_configuration[0],
                "segment-size": best_configuration[1],
                "est_tp": best_obj[0],
                "est_l": best_obj[1],
                "est_cpu": best_obj[2]
            }

            logger.info(f"Best configuration for {operator['id']}: {configurations[operator['id']]}.")

        # Count occurrences of each segment size across all operators
        count_segment_size = {16384: 0, 32768: 0, 65536: 0}
        for operator in configurations.keys():
            configuration = configurations[operator]
            for _ in range(int(configuration["parallelism"])):
                count_segment_size[configuration["segment-size"]] = count_segment_size[configuration["segment-size"]] + 1

        # Select the most common segment size across all configurations for the pipeline
        pipeline_segment_size = max(count_segment_size, key=count_segment_size.get)

        # Update all configurations to use the selected segment size
        for operator in configurations.keys():
            configuration = configurations[operator]
            configuration["segment-size"] = pipeline_segment_size

        return configurations

    def schedule(self, configurations, resources):
        scheduler = Scheduler()
        operators_to_schedule = {16384: {}, 32768: {}, 65536: {}}
        total_req_cpu = {16384: 0.0, 32768: 0.0, 65536: 0.0}

        # Distribute operators based on segment size and sum CPU requirements
        for operator in configurations.keys():
            configuration = configurations[operator]
            total_req_cpu[configuration["segment-size"]] = total_req_cpu[configuration["segment-size"]] + configuration["est_cpu"]
            operators_to_schedule[configuration["segment-size"]] = {**operators_to_schedule[configuration["segment-size"]], **{operator: configuration}}

        # Prioritize scheduling by total CPU demand per segment size
        priority_segment_size = sorted(total_req_cpu, key=total_req_cpu.get, reverse=True)

        # Schedule operators
        placement_full = {}
        resources_full = {}
        cost = 0
        for segment_size in priority_segment_size:

            # Check if there are operators to schedule
            if operators_to_schedule[segment_size] != {}:
                placement, resources, _ = scheduler.schedule(operators_to_schedule[segment_size], resources, segment_size)

                # Update placement and resource allocations
                for p in placement:
                    if not placement[p] == []:
                        placement_full[p] = placement[p]
                for r in resources:
                    resources_full[r] = resources[r]

                # Accumulate scheduling cost
                cost = cost + scheduler.calc_cost(placement, resources)

        return placement_full, resources_full, cost, operators_to_schedule
