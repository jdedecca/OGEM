__author__ = 'joaogorenstein'

import sys, copy, os, warnings
from OGEM import Load_Parameters, Create_Output_Files, Join_XLS
from OGEM import Load_Parameters as Load_Parameters_Old
from itertools import product
import matplotlib.pyplot as plt
import multiprocessing
from shutil import copyfile
from subprocess import call


# @profile
#print(mem_top())

def run_simulation(data):
    """ Calls each run for a given run schedule.
        Runs are executed through the shell because of a PyPSA memory leak.
    """

    run_number = data[0]
    sens_parameters = data[1]
    simulation_file = data[2]

    run_number = str(run_number)
    sens_parameters = str(sens_parameters)
    call(["python", "simulation.py", run_number, sens_parameters, simulation_file])


def schedule1():
    """" Creates output files, sets up multiprocessing and calls runs """

    #Set-up
    output = []
    plt.clf() # Clear mpyplot figures.

    f = []
    for (dirpath, dirnames, filenames) in os.walk("Simulations"):
        f.extend(filenames)

    simulation_file = os.path.join("Simulations", f[0])
    print(simulation_file) # Indicate current run schedule.

    parameters = Load_Parameters(simulation_file)

    number_runs = 1
    for parameter in parameters["sens_values"]:
        number_runs = number_runs * len(list(parameter))

    Create_Output_Files(f) # Stores parameters for all run schedules and creates output file for data.

    for file in f: # Runs of a single run schedule can be parallel. Separate run schedules are executed sequentially.

        if file[:4] == "OGEM": # Run schedules must start with "OGEM".
            data = []
            simulation_file = os.path.join("Simulations", file)
            print("Running", file)
            parameters = Load_Parameters(simulation_file)

            for run_number, sens_parameters in enumerate(product(*parameters["sens_values"])):
                data.append((run_number, sens_parameters, simulation_file))

            p = multiprocessing.Pool(1) # Single run for debugging.
            #p = multiprocessing.Pool(multiprocessing.cpu_count())  # Parallel run.
            p.map(run_simulation, data)

            Join_XLS(number_runs) # Joins data of all runs for the current run schedule.

    return output

if __name__ == '__main__':
    schedule1()

