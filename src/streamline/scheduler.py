from itertools import combinations
import logging

# Create a logger instance
logger = logging.getLogger(__name__)


class Scheduler:

    def filter_resources_based_on_segment_size(self, resources, segment_size):

        # Filter resources based on whether their segment size matches the required segment size
        filtered = {}
        for r in resources.keys():

            # Include resource if no slots are occupied or if it matches the required segment size
            if resources[r]["occupied_slots"] == 0 or ("segment-size" in resources[r] and resources[r]["segment-size"] == segment_size):
                filtered[r] = resources[r]

        return filtered

    def find_minimum_subset(self, arr, target):

        # Sort the resources by their CPU time
        arr = sorted(arr)
        best_subset = None
        best_sum = float('inf')
        for r in range(1, len(arr) + 1):

            # Try all combinations of r elements
            for subset in combinations(arr, r):

                # Calculate the sum of the combination
                subset_sum = round(sum(subset), 1)

                # If the sum meets the target
                if subset_sum >= target:

                    # Find the combination with the minimum sum that is still >= target
                    if subset_sum < best_sum:
                        best_subset = subset
                        best_sum = subset_sum

                    # Early exit for this combination length (r)
                    break

        # Return the best subset or empty list
        return list(best_subset) if best_subset else []

    def calc_cost(self, placement, resources):
        cost = 0.0

        # Iterate over placements
        for r in placement:

            # Check if resource contains operators
            if len(placement[r]) > 0:
                cost = cost + resources[r]["cost"]

        return cost

    def get_operator_data(self, operators):
        sum_all_cpu = 0.0
        cpu_est_operator = {}
        segment_size_operator = {}

        # Retrieve the CPU and segment size data for the operators
        for op in operators.keys():
            for i in range(0, int(operators[op]["parallelism"])):
                cpu_est_operator[op + "_" + str(i)] = operators[op]["est_cpu"] / operators[op]["parallelism"]
                segment_size_operator[op + "_" + str(i)] = operators[op]["segment-size"]
                sum_all_cpu = sum_all_cpu + cpu_est_operator[op + "_" + str(i)]

        return sum_all_cpu, cpu_est_operator, segment_size_operator

    def schedule_on_resources(self, cpu_est_operator, selected_resources, resource_cpu_available, placement, resource_ids, resources, segment_size):
        # Sort operators by CPU requirement
        sorted_operators = sorted(cpu_est_operator.items(), key=lambda x: x[1], reverse=True)

        for operator, requirement in sorted_operators:
            best_resource_index = 0
            best_score = 0

            # Iterate over resources (try to fill up all available resources equally)
            sorted_resources = sorted(range(len(selected_resources)), key=lambda x: resource_cpu_available[x], reverse=True)
            for i in sorted_resources:

                # Check if resource fulfills requirements and select best resource based on score
                if resource_cpu_available[i] >= requirement:
                    score = resource_cpu_available[i]
                    if score > best_score:
                        best_score = score
                        best_resource_index = i

            # Update resource status after placing the operator
            resource_cpu_available[best_resource_index] -= requirement
            placement[resource_ids[best_resource_index]].append(operator)
            resources[resource_ids[best_resource_index]]["occupied_slots"] = resources[resource_ids[best_resource_index]]["occupied_slots"] + 1
            resources[resource_ids[best_resource_index]]["load"] = resources[resource_ids[best_resource_index]]["load"] + requirement
            resources[resource_ids[best_resource_index]]["segment-size"] = segment_size

        return placement, resources

    def schedule(self, operators, resources, segment_size):
        utilization = 0.8
        placement = {r: [] for r in resources}

        usable_cpu_on_already_used_resources = 0
        for r in resources:

            # Check if the resource is already in use and has a matching segment size
            if resources[r]["occupied_slots"] > 0 and resources[r]["segment-size"] == segment_size:

                available_load_percent = (resources[r]["cores"] * utilization - resources[r]["load"]) / resources[r]["cores"]

                # Only use already used resources if more than 10% is available (to avoid overutilization)
                if available_load_percent > 0.1:
                    usable_cpu_on_already_used_resources = usable_cpu_on_already_used_resources + available_load_percent

        logger.info("Usable CPU time of already used resources: " + str(usable_cpu_on_already_used_resources))

        # Calculate total required CPU time for operators
        sum_all_cpu, cpu_est_operator, segment_size_operator = self.get_operator_data(operators)
        sum_all_cpu = sum_all_cpu - usable_cpu_on_already_used_resources
        logger.info("Total Required CPU time " + str(sum_all_cpu))

        # Filter resources based on segment size
        filtered = self.filter_resources_based_on_segment_size(resources, segment_size)

        # Prepare sorted resources
        sorted_resources = sorted(filtered.items(), key=lambda x: x[1]['occupied_slots'], reverse=True)
        available_resource_cores = {}
        available_resource_cores_only = []
        for r_id, details in sorted_resources:
            if details["occupied_slots"] == 0:
                if not details["cores"] in available_resource_cores:
                    available_resource_cores[details["cores"]] = []
                available_resource_cores[details["cores"]].append(r_id)
                available_resource_cores_only.append(details["cores"])

        # Calculate maximum CPU usage per resource
        available_resource_max_cpu_usage = [cores * utilization for cores in available_resource_cores_only]

        # Find and select minimum subset of resources for the given cpu-time
        selected_resources = [cores * (1.0 / utilization) for cores in self.find_minimum_subset(available_resource_max_cpu_usage, sum_all_cpu)]
        resource_ids = []
        for r in selected_resources:
            resource_ids.append(available_resource_cores[r][0])
            available_resource_cores[r].remove(available_resource_cores[r][0])

        # Schedule operators on resources
        resource_cpu_available = [cores * utilization for cores in selected_resources]

        resources_already_used_with_segment_size = {}
        for r in resources:
            if resources[r]["occupied_slots"] > 0 and resources[r]["segment-size"] == segment_size:
                available_load_percent = (resources[r]["cores"] * utilization - resources[r]["load"]) / resources[r]["cores"]
                if available_load_percent > 0.1:
                    selected_resources.append(resources[r]["cores"])
                    resource_ids.append(r)
                    resource_cpu_available.append(resources[r]["cores"] * utilization - resources[r]["load"])

        # Final placement of operators on resources
        placement, resources = self.schedule_on_resources(cpu_est_operator, selected_resources, resource_cpu_available, placement, resource_ids, resources, segment_size)
        details = [selected_resources, resource_cpu_available, placement, resource_ids, resources]

        return placement, resources, details
