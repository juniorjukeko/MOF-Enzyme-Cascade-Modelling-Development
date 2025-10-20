# visualization.py
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.ticker import ScalarFormatter, FuncFormatter
matplotlib.use('TkAgg')

def plot_enzyme_pore_profiles(model, save_path=None):
    """
    Plot EA and EB profiles along the pore length (x from 0 to L)
    
    Parameters:
    model: **Solved** Pyomo model
    save_path: If provided, save plot to this path
    """
    # Extract x values from the discretization
    x_values = sorted(list(model.x))
    
    # Extract enzyme profiles
    EA_values = [model.EA_x_profile[x]() for x in x_values]
    EB_values = [model.EB_x_profile[x]() for x in x_values]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(x_values, EA_values, 'r-', linewidth=4, label='Enzyme A')
    ax.plot(x_values, EB_values, 'b-', linewidth=4, label='Enzyme B')
    
    ax.set_xlabel('Pore Position x (dm)', fontsize=16)
    ax.set_ylabel('Enzyme Surface Density (μmol/dm²)', fontsize=16)
    ax.set_title('Enzyme Density Distribution Along Pore Length', fontsize=18)
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3)
    
    # Set x limits from 0 to L
    ax.set_xlim(0, model.L.value)
    
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Convert axis to scientific notation
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    # ax.xaxis.get_major_formatter().set_powerlimits((-3, 3))
    ax.ticklabel_format(axis='x', style='scientific', scilimits=(-3, 3))
    
    # Put max enzyme densities to the plot
    EA_max = max(EA_values)
    EB_max = max(EB_values)
    ax.text(0.02, 0.48, f'EA max: {EA_max:.1f}\nEB max: {EB_max:.1f}', 
            transform=ax.transAxes, verticalalignment='top', fontsize=16,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Enzyme profile plot saved to: {save_path}")
    
    plt.show()
    
    return fig

def plot_substrate_time_profiles(model, save_path=None):
    """
    Plot S1, S2, S3 concentrations over reaction time (t from 0 to tf)
    
    Parameters:
    model: **Solved** Pyomo model
    save_path: If provided, save plot to this path
    """
    # Extract time values from the discretizied parameter
    t_values = sorted(list(model.time))
    
    # Extract substrate concentrations
    S1_values = [model.S_0['S1', t]() for t in t_values]
    S2_values = [model.S_0['S2', t]() for t in t_values] 
    S3_values = [model.S_0['S3', t]() for t in t_values]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(t_values, S1_values, 'k-', linewidth=4, label='S1 (Substrate)')
    ax.plot(t_values, S2_values, 'b-', linewidth=4, label='S2 (Intermediate)')
    ax.plot(t_values, S3_values, 'r-', linewidth=4, label='S3 (Product)')
    
    ax.set_xlabel('Reaction time t (min)', fontsize=16)
    ax.set_ylabel('Concentration (μM)', fontsize=16)
    ax.set_title('Substrate Concentrations Over Reaction Time', fontsize=18)
    ax.legend(fontsize=16)
    ax.grid(False)
    
    # Set x limits from 0 to tf
    ax.set_xlim(0, model.tf.value)
    
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Add final concentration annotations
    final_S1 = S1_values[-1]
    final_S2 = S2_values[-1] 
    final_S3 = S3_values[-1]
    
    ax.text(0.48, 0.98, f'Final concentrations:\nS1: {final_S1:.2f} μM\nS2: {final_S2:.2f} μM\nS3: {final_S3:.2f} μM',
            transform=ax.transAxes, verticalalignment='top', fontsize=16,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Substrate concentration plot saved to: {save_path}")
    
    plt.show()
    
    return fig

