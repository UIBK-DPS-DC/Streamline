import queue
import logging

# Create a logger instance
logger = logging.getLogger(__name__)


class Pipeline:

    def __init__(self, name, operators, edges, sources, sinks):
        self.name = name
        self.operators = operators
        self.edges = edges
        self.sources = sources
        self.sinks = sinks

    def list_vertices(self):
        return self.operators

    def get_structure(self):

        # Create the graph structure with nodes, edges, sources and sinks
        graph = {
            "operators": [operator for operator in self.operators],
            "edges": [{"from": edge[0], "to": edge[1]} for edge in self.edges],
            "sources": [source for source in self.sources],
            "sinks": [sink for sink in self.sinks]
        }
        return graph

    def get_successors(self, operator):
        successors = []
        for source, dest in self.edges:
            if operator == source:
                successors.append(dest)
        return successors

    def get_sources(self):
        return self.sources

    def get_sinks(self):
        return self.sinks

    def get_predecessors(self, operator):
        predecessors = []
        for source, dest in self.edges:
            if operator == dest:
                predecessors.append(source)
        return predecessors

    def get_load(self, input_load, data):
        load = {}

        # Create a queue for processing operators
        q = queue.Queue()

        # Start with sources in the queue
        for source in self.sources:
            q.put(source)
            load[source] = input_load

        while not q.empty():

            # Get an operator from the queue
            operator = q.get()

            # Get successors for the operator
            successors = [last for first, last in self.edges if first == operator]
            for s in successors:
                # Add successor to the queue for processing
                q.put(s)

                # Calculate the load for the successor
                load[s] = load[operator] * data[operator]

        return load
