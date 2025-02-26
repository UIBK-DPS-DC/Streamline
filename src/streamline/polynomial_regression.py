import numpy
import warnings
import numpy as np
import logging

# Create a logger instance
logger = logging.getLogger(__name__)

# Suppress RankWarning from numpy (for polynomial fitting)
warnings.filterwarnings("ignore", category=np.RankWarning)


class PolynomialRegression:

    def __init__(self, data, create_plots=False):
        self.data = data
        self.create_plots = create_plots

    def train(self, x, y, operator_type):

        # Scale y values
        y = numpy.array(y) * 0.8

        # Fit a cubic polynomial model to the data
        model_ = numpy.poly1d(numpy.polyfit(x, y, 3))

        # If plot creation is enabled
        if self.create_plots:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(x, y, label='y = f(x)', color='blue', marker='o', linestyle='-', linewidth=2)
            file = "./polynomial_regeression_" + str(operator_type) + ".png"
            plt.savefig(file)
            logger.info("Finished training for " + str(operator_type) + ": " + file)

        return model_

    def predict(self, model_, value):
        # Make predictions using the trained polynomial model
        return model_(value)

    def generate_models_for_operators(self):
        data = self.data

        # Get unique operator types from the data
        operator_types = data['Operator Type'].unique()
        poly_reg_models = {}
        for operator_type in operator_types:
            x = []
            y = []

            # Filter data by operator type
            filtered_df = data[(data["Operator Type"] == operator_type)]

            # Get unique parallelism values
            unique_parallelism = np.sort(filtered_df["Parallelism"].unique())

            # Skip if there is only one unique parallelism value
            if not len(unique_parallelism) > 1:
                continue

            # Iterate over all parallelism values
            for parallelism in unique_parallelism:

                # Filter data for current parallelism value
                filter_p = filtered_df[(filtered_df["Parallelism"] == parallelism)]
                avg_tp_in = 0

                # Avoid division by zero
                avg_busy_time = 0.001
                for index, row in filter_p.iterrows():
                    avg_tp_in = avg_tp_in + row["TP_in"]
                    avg_busy_time = avg_busy_time + row["Busytime"]
                num_rows = filter_p.shape[0]

                # Calculate average TP_in
                avg_tp_in = avg_tp_in / num_rows

                # Calculate average busy time
                avg_busy_time = avg_busy_time / num_rows

                x.append(parallelism)
                y.append(avg_tp_in / avg_busy_time)

            # Train model for operator type
            poly_reg_models[operator_type] = self.train(x, y, operator_type)

        return poly_reg_models
