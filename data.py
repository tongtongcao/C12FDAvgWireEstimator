import numpy as np

def read_file(filename):
    """
    Reads one CSV file
    """
    events = []
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip() != ""]  # remove empty lines

    for line in lines:
        averageWires = np.array([float(x) for x in line.split(",")] )
        events.append(averageWires)
    return events