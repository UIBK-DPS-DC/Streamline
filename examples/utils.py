import sys

sys.path.append('src')
from streamline import Pipeline


def get_pipeline_1():
    data = {
        "operators": [
            {"id": "v_0", "label": "source", "type": 0},
            {"id": "v_1", "label": "map", "type": 1},
            {"id": "v_2", "label": "filter", "type": 2},
            {"id": "v_3", "label": "sink", "type": 3}
        ],
        "edges": [
            ("v_0", "v_1"),
            ("v_1", "v_2"),
            ("v_2", "v_3")
        ],
        "sources": ["v_0"],
        "sinks": ["v_3"]
    }
    return Pipeline("name", data["operators"], data["edges"], data["sources"], data["sinks"])


def get_pipeline_2():
    data = {
        "operators": [
            {"id": "v_5", "label": "source", "type": 5},
            {"id": "v_6", "label": "map", "type": 6},
            {"id": "v_7", "label": "sink", "type": 7}
        ],
        "edges": [
            ("v_5", "v_6"),
            ("v_6", "v_7")
        ],
        "sources": ["v_5"],
        "sinks": ["v_7"]
    }
    return Pipeline("name", data["operators"], data["edges"], data["sources"], data["sinks"])


def get_resources():
    # example resource
    return {
        "R1": {
            "occupied_slots": 0,
            "cores": 4,
            "load": 0.0,
            "cost": 0.192
        },
        "R2": {
            "occupied_slots": 0,
            "cores": 4,
            "load": 0.0,
            "cost": 0.192
        },
        "R3": {
            "occupied_slots": 0,
            "cores": 4,
            "load": 0.0,
            "cost": 0.192
        },
        "R4": {
            "occupied_slots": 0,
            "cores": 4,
            "load": 0.0,
            "cost": 0.192
        },
        "R5": {
            "occupied_slots": 0,
            "cores": 4,
            "load": 0.0,
            "cost": 0.192
        },
        "R6": {
            "occupied_slots": 0,
            "cores": 4,
            "load": 0.0,
            "cost": 0.192
        },
        "R7": {
            "occupied_slots": 0,
            "cores": 4,
            "load": 0.0,
            "cost": 0.192
        },
        "R8": {
            "occupied_slots": 0,
            "cores": 4,
            "load": 0.0,
            "cost": 0.192
        },
        "R9": {
            "occupied_slots": 0,
            "cores": 4,
            "load": 0.0,
            "cost": 0.192
        },
        "R10": {
            "occupied_slots": 0,
            "cores": 4,
            "load": 0.0,
            "cost": 0.192
        },
        "R11": {
            "occupied_slots": 0,
            "cores": 8,
            "load": 0.0,
            "cost": 0.384
        },
        "R12": {
            "occupied_slots": 0,
            "cores": 8,
            "load": 0.0,
            "cost": 0.384
        },
        "R13": {
            "occupied_slots": 0,
            "cores": 8,
            "load": 0.0,
            "cost": 0.384
        },
        "R14": {
            "occupied_slots": 0,
            "cores": 8,
            "load": 0.0,
            "cost": 0.384
        },
        "R15": {
            "occupied_slots": 0,
            "cores": 8,
            "load": 0.0,
            "cost": 0.384
        },
        "R16": {
            "occupied_slots": 0,
            "cores": 8,
            "load": 0.0,
            "cost": 0.384
        },
        "R17": {
            "occupied_slots": 0,
            "cores": 8,
            "load": 0.0,
            "cost": 0.384
        },
        "R18": {
            "occupied_slots": 0,
            "cores": 8,
            "load": 0.0,
            "cost": 0.384
        },
        "R19": {
            "occupied_slots": 0,
            "cores": 8,
            "load": 0.0,
            "cost": 0.384
        },
        "R20": {
            "occupied_slots": 0,
            "cores": 8,
            "load": 0.0,
            "cost": 0.384
        },
        "R21": {
            "occupied_slots": 0,
            "cores": 16,
            "load": 0.0,
            "cost": 0.768
        },
        "R22": {
            "occupied_slots": 0,
            "cores": 16,
            "load": 0.0,
            "cost": 0.768
        },
        "R23": {
            "occupied_slots": 0,
            "cores": 16,
            "load": 0.0,
            "cost": 0.768
        },
        "R24": {
            "occupied_slots": 0,
            "cores": 16,
            "load": 0.0,
            "cost": 0.768
        },
        "R25": {
            "occupied_slots": 0,
            "cores": 16,
            "load": 0.0,
            "cost": 0.768
        },
        "R26": {
            "occupied_slots": 0,
            "cores": 16,
            "load": 0.0,
            "cost": 0.768
        },
        "R27": {
            "occupied_slots": 0,
            "cores": 16,
            "load": 0.0,
            "cost": 0.768
        },
        "R28": {
            "occupied_slots": 0,
            "cores": 16,
            "load": 0.0,
            "cost": 0.768
        }
    }
