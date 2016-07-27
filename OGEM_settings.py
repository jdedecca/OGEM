__author__ = 'joaogorenstein'

def init():
    """ Global variables, with data being the most important """
    global data
    global base_kV
    global base_MVA
    global base_Z

    base_kV = 400
    base_MVA = 100
    base_Z = base_kV ** 2 / (base_MVA * 1000)
