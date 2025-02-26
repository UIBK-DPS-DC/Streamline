import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.core.population import Population
import logging

# Create a logger instance
logger = logging.getLogger(__name__)


class Problem(ElementwiseProblem):

    def __init__(self, forecast, model, type, max_parallelism, poly_reg_models, pred_parallelism):
        self.forecast = forecast
        self.model = model
        self.type = type
        self.max_parallelism = max_parallelism
        self.poly_reg_models = poly_reg_models
        self.pred_parallelism = pred_parallelism

        # Define the optimization problem with two variables (parallelism, segment size)
        # and three objectives (maximize throughput, minimize latency, minimize CPU usage)
        super().__init__(n_var=2, n_obj=3, n_ieq_constr=0, xl=[1, 16_000], xu=[self.max_parallelism, 64_000])

    def _evaluate(self, x, out, *args, **kwargs):
        # Evaluate the given configuration
        throughput, latency, cpu = self.evaluate_configuration(x)

        # Use negative throughput to maximize it (since pymoo minimizes objectives)
        out["F"] = [-throughput, latency, cpu]

    def check_future_load(self, parallelism, segment_size):
        sum_throughput = 0
        sum_latency = 0
        sum_cpu = 0

        # Iterate of future workload prediction
        for future_value in self.forecast:
            # Use polynomial regression model if available, otherwise default to future_value (for sources)
            pa_current = self.poly_reg_models[self.type](parallelism) if self.type in self.poly_reg_models else future_value

            # The system can only handle a limited input, capped by available parallelism
            load_input = min(future_value, pa_current)

            # Prepare input data for Neural Network model prediction
            input_data = np.array([[load_input, self.pred_parallelism, parallelism, segment_size, self.type]])

            # Predict latency and cpu requirements
            predicted_latency, predicted_cpu = self.model.predict(input_data)[0]

            # Sum up metrics over the forecasted period
            sum_throughput = sum_throughput + load_input
            sum_latency = sum_latency + predicted_latency
            sum_cpu = sum_cpu + predicted_cpu

        # Compute average values over the forecast period
        avg_throughput = sum_throughput / len(self.forecast)
        avg_latency = sum_latency / len(self.forecast)
        avg_cpu = sum_cpu / len(self.forecast)

        return avg_throughput, avg_latency, avg_cpu

    def evaluate_configuration(self, config):
        # Ensure configuration values are valid before evaluation
        config = self.repair(config)

        # Check config for a range of future load values
        avg_throughput, avg_latency, avg_cpu = self.check_future_load(config[0], config[1])

        # return results of configuration evaluation
        return avg_throughput, avg_latency, avg_cpu

    def repair(self, config):
        # Adjust parallelism to the closest valid value
        config[0] = self.find_closest(list(range(1, self.max_parallelism + 1)), config[0])

        # Adjust segment size to the closest valid value
        config[1] = self.find_closest([16_384, 32_768, 65_536], config[1])

        return config

    def find_closest(self, lst, target):
        # Find the closest valid value to the target in a given list
        return min(lst, key=lambda x: abs(x - target))


class ConfigurationSelector:

    def __init__(self, forecast, o_type, model, poly_reg_models, pred_parallelism):
        self.forecast = forecast
        self.o_type = o_type
        self.model = model
        self.res = None
        self.problem = None
        self.poly_reg_models = poly_reg_models
        self.pred_parallelism = pred_parallelism

    def solve(self, max_parallelism):
        # Create an optimization problem instance
        self.problem = Problem(self.forecast, self.model, self.o_type, max_parallelism, self.poly_reg_models, self.pred_parallelism)

        # Initialize the population with custom values
        initial_population = Population.new("X", self.create_population())

        # Run NSGA-II for 20 generations
        algorithm = NSGA2(pop_size=20, initialization=initial_population)
        self.res = minimize(self.problem, algorithm, ("n_gen", 20), verbose=False, seed=1)

    def get_best(self, weights):
        min_values = self.res.F.min(axis=0)
        max_values = self.res.F.max(axis=0)
        range_vals = max_values - min_values

        # Avoid division by zero
        range_vals[range_vals == 0] = 1

        # Normalize the objective function values
        normalized_f = (self.res.F - min_values) / range_vals

        # Compute weighted scores
        weighted_scores = np.dot(normalized_f, np.array(weights))

        # Select the best configuration
        best_index = np.argmin(weighted_scores)
        best_configuration = self.problem.repair(self.res.X[best_index])
        best_score = weighted_scores[best_index]

        # Evaluate the best configuration
        avg_throughput, avg_latency, avg_cpu = self.problem.check_future_load(best_configuration[0], best_configuration[1])

        return best_configuration, best_score, [avg_throughput, avg_latency, avg_cpu]

    def create_population(self):
        num_individuals = 99
        population = np.zeros((num_individuals, 3))

        for i in range(num_individuals):
            # Assign a random parallelism level
            population[i, 0] = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8])

            # Assign a random segment size within the given range
            population[i, 1] = np.random.randint(16_000, 64_0000)

        return population
