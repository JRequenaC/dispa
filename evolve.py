""" My evolve.py docstring  """

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib import animation, rc
from IPython.display import HTML


def get_APD(DI):
    tauAPD = 0.025
    cAPD = 0.140
    return cAPD*(1-np.exp(-DI/tauAPD))


def get_CV(DI):
    tauCV  = 0.035
    cVC = 60
    return cVC*(1-np.exp(-DI/tauCV))


def get_excitation_probability(tissue_state, rest_cells, t):
    cell_excitability       = get_CV((t-tissue_state['t repol']))*rest_cells
    tissue_dim = int(np.sqrt(cell_excitability.size))
    macro_state = tissue_state['Macro state'].values.reshape(tissue_dim,tissue_dim) 
    diffusion_kernel = np.array([[ 0, 1, 0],[1, 0, 1],[ 0, 1, 0]])
    cell_excitation         = signal.convolve2d(macro_state*(macro_state==2), diffusion_kernel, boundary='fill', mode='same')
    excitation_probability   = 0.001*cell_excitation.flatten()*cell_excitability
    return excitation_probability


def get_internally_stimulated_cells(tissue_state, rest_cells, t):
    total_cells                      = tissue_state.shape[0]
    excitation_probability           = get_excitation_probability(tissue_state, rest_cells, t)
    internally_stimulatedCells       = rest_cells & (np.random.rand(total_cells) < excitation_probability)
    return internally_stimulatedCells


def update_stimulus_time(applied_stimulus):
    applied_stimulus.stimulus_index +=1
    tissue_dim = int(np.sqrt(applied_stimulus.affected_cells.size))
    x, y= np.mgrid[0:tissue_dim, 0:tissue_dim]
    applied_stimulus.affected_cells = (y < tissue_dim/2)&(x < tissue_dim/2)


def get_externally_stimulated_cells(applied_stimulus, rest_cells, t):
    externally_stimulated_cells = np.zeros(rest_cells.size, dtype=bool)
    if applied_stimulus.stimulation_times[applied_stimulus.stimulus_index] < t:        
        externally_stimulated_cells = rest_cells & applied_stimulus.affected_cells.flatten()
        update_stimulus_time(applied_stimulus)
    return externally_stimulated_cells


def get_cell_transitions_from_rest(tissue_state, applied_stimulus, t):    
    rest_cells                       = (tissue_state['Macro state']==0)
    internally_stimulated_cells       = get_internally_stimulated_cells(tissue_state, rest_cells, t)
    expternally_stimulated_cells      = get_externally_stimulated_cells(applied_stimulus, rest_cells, t)
    rest_to_excited_cell_transitions    = internally_stimulated_cells | expternally_stimulated_cells
    return rest_to_excited_cell_transitions


def get_cell_transitions_from_excited(tissue_state, t):
    fraction_APD   = 0.2
    excited_cells                        = (tissue_state['Macro state']==2)
    excited_to_refractory_cell_transitions  = excited_cells & ((t-tissue_state['t depol'])>(fraction_APD*tissue_state['APD']))
    return excited_to_refractory_cell_transitions


def get_cell_transitions_from_refractory(tissue_state, t):
    refractory_cells                     = (tissue_state['Macro state']==1)
    refractory_to_rest_cell_transitions     = refractory_cells & ((t-tissue_state['t depol'])>tissue_state['APD'])
    return refractory_to_rest_cell_transitions


def change_tissue_state(tissue_state, applied_stimulus, t):

    rest_to_excited_cell_transitions            = get_cell_transitions_from_rest(tissue_state, applied_stimulus, t)
    excited_to_rerfactory_cell_transitions      = get_cell_transitions_from_excited(tissue_state, t)
    refractory_to_rest_cell_transitions         = get_cell_transitions_from_refractory(tissue_state, t)

    tissue_state['Macro state'] = tissue_state['Macro state'] + 2*rest_to_excited_cell_transitions-refractory_to_rest_cell_transitions-excited_to_rerfactory_cell_transitions 
    tissue_state['t repol']  = t*refractory_to_rest_cell_transitions+tissue_state['t repol']*(~refractory_to_rest_cell_transitions)
    tissue_state['t depol'] = t*rest_to_excited_cell_transitions+tissue_state['t depol']*(~rest_to_excited_cell_transitions)   
    tissue_state['APD'] = get_APD((tissue_state['t depol']-tissue_state['t repol']))*rest_to_excited_cell_transitions + tissue_state['APD']*(~rest_to_excited_cell_transitions)

    return tissue_state

class StimulationProtocol:
    
    stimulus_index = 0
    stimulation_times = np.array([0., 0.3, 10.])
    
    def __init__(self, tissue_dim):
        x, y= np.mgrid[0:tissue_dim, 0:tissue_dim]
        self.affected_cells = (y == 0)


def animate(n):
    global df, t
    df = change_tissue_state(df, applied_stimulus, t)
    s = df['Macro state'].values.reshape(TISSUE_DIM,TISSUE_DIM)
    voltage_index=np.clip(np.int16(round((t-df['t depol'])*1800/df['APD'])),0,4999)
    voltage = action_potential[voltage_index]
    im.set_array(voltage.reshape(TISSUE_DIM, TISSUE_DIM))
    plt.title(format(t, '.5f'))
    t += delta_t
    return im,



if __name__ == "__main__":

    TISSUE_DIM = 150
    NUMBER_CELLS = TISSUE_DIM*TISSUE_DIM
    delta_t = 0.001
    t = 0

    action_potential = np.load('action_potential_shape.npy')
    d = {'Macro state': np.zeros(NUMBER_CELLS), 't depol': -0.4*np.ones(NUMBER_CELLS), 't repol': -0.2*np.ones(NUMBER_CELLS), 'APD': 0.2*np.ones(NUMBER_CELLS)}
    df = pd.DataFrame(d)
    s = df['Macro state'].values.reshape(TISSUE_DIM, TISSUE_DIM)

    applied_stimulus = StimulationProtocol(TISSUE_DIM)
    fig = plt.figure()
    im = plt.imshow(s,  cmap = 'jet', 
                vmin=-84, vmax=45,
                animated = True)
    plt.colorbar()
    anim = animation.FuncAnimation(fig, animate, frames=100,
                               interval=10, blit=True)
                    
    plt.show()



