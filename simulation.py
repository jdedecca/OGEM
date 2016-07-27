__author__ = 'joaogorenstein'

import sys, copy, warnings, gc
from OGEM import Period_Run, PyPSA_Network_Setup, Load_Parameters, Load_Data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import OGEM_settings
from shutil import copyfile
import pypsa


def main():
    """ Executes a single run of the exploratory model """

    pd.set_option("precision", 3)
    pd.set_option("max_rows", 400)
    pd.set_option("max_columns", 200)
    pd.set_option('display.width', 200)
    pd.set_option('expand_frame_repr', False)
    np.set_printoptions(precision=3)

    OGEM_settings.init() # Initializes the data global variable
    # Retrieves the run parameters passed through the shell command
    run_number = sys.argv[1]
    sens_parameters = sys.argv[2][1:len(sys.argv[2])-1]
    simulation_file = sys.argv[3]
    sens_parameters = [float(x) for x in sens_parameters.split(',')] # Parameters for current run
    parameters = Load_Parameters(simulation_file)

    OGEM_settings.data = Load_Data(simulation_file) # Loads setup data
    parameters["run_number"] = run_number
    selected_projects = {}
    output = []

    selection_data = []
    parameters["current_parameters_value"] = sens_parameters

    print("Run number", run_number)
    print("Parameters:", parameters["current_parameters_value"])

    # Network setup
    pypsa_network = pypsa.Network()
    links_dataframe, pypsa_network = PyPSA_Network_Setup(0, pypsa_network)
    parameters["number of links"] = pypsa_network.lines.shape[0] + pypsa_network.transport_links.shape[0]

    # Update parameters for current run.
    for i, value in enumerate(parameters["current_parameters_value"]):
        try:
            parameters[parameters["sens_name"][i]]
        except:
            print("Error: parameter", parameters["sens_name"][i], "does not exist")
            sys.exit()
        parameters[parameters["sens_name"][i]] = value


    # Main loop for expansion simulation.
    for period in range(parameters["periods"]):

        selected_projects, selection_data, pypsa_network, links_dataframe = Period_Run(run_number, period, selection_data, pypsa_network, links_dataframe)

        output.append([period, selected_projects])

    for x in range(len(output)):
        print("Period", [x][0], "expansions:", output[x][1])

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    if parameters["interactive"] != -1:
        plt.ioff()
        plt.show()

if __name__=='__main__':
    main()