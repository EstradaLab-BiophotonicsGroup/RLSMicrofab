import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statistics import median
from matplotlib import cm

Path = r'D:/JøFrå/Estudio/Facultad/Fisica/'

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FUNCTIONS

NumberOfExperiments = 2

def LoadData(NumberOfExperiments, Path = Path):
    Experiments = {}
    for n in range(1, NumberOfExperiments + 1):
        Data = pd.read_csv(Path+'/Datos/Experimento {} con WT.csv'.format(n), delimiter=';', index_col='ID')
        Experiments['No. {}'.format(n)] = Data
    return Experiments

def LoadCalcoFluorData(NumberOfExperiments, Path = Path):
    Experiments = {}
    for n in range(1, NumberOfExperiments + 1):
        Data = pd.read_csv(Path+'/Datos/Resultados de Experimento {} calcofluor en chip.csv'.format(n), delimiter=';', index_col='ID')
        Data.drop(columns=['Unnamed: 2'], inplace=True)
        Experiments['No. {}'.format(n)] = Data
    return Experiments

def RandomSelector(CellList, NumberOfCells):
    try:
        CellsToRemove = random.sample(range(0, len(CellList)), NumberOfCells)
        NewList = [0] * NumberOfCells
        for n, j in enumerate(CellsToRemove):
            NewList[n] = CellList[j]
    except ValueError:
        print('You requested to keep more cells than available in the list.')
    return NewList

Experiments = LoadData(NumberOfExperiments)

CalcoFluor = LoadCalcoFluorData(NumberOfExperiments)

Experiments['All Together'] = pd.concat([Experiments['No. 1'], Experiments['No. 2']], ignore_index=True)

CalcoFluor['All Together'] = pd.concat([CalcoFluor['No. 1'], CalcoFluor['No. 2']], ignore_index=True)

#%%########################################################################### MANIPULATED DATA

#%%%########################################################################## Data Trimming

TotalIterations = 100
IterationsPerDistribution = 100
FinalMedians = {}

# Iterate over each experiment
for Experiment in Experiments.keys():
    PreviousMothersCount = len(CalcoFluor[Experiment]['Scars_finales'])    
    PreviousHeights = np.histogram(CalcoFluor[Experiment]['Scars_finales'], bins=int(max(CalcoFluor[Experiment]['Scars_finales']) - min(CalcoFluor[Experiment]['Scars_finales'])))[0]
    PreviousScars = np.histogram(CalcoFluor[Experiment]['Scars_finales'], bins=int(max(CalcoFluor[Experiment]['Scars_finales']) - min(CalcoFluor[Experiment]['Scars_finales'])))[1]
    CombinedHistograms = []
    m = 0
    Medians = []
    CutMedians = []
    # Outer loop for total iterations
    while m < TotalIterations:
        m += 1
        print(f'\rIteration {m} ► {TotalIterations}', end=' ')
        RandomList = np.array(RandomSelector(list(Experiments[Experiment]['Generations']), PreviousMothersCount))
        CutMedians.append(int(median(RandomList)))
        # Inner loop for iterations per distribution
        i = 0
        while i < IterationsPerDistribution:
            i += 1
            random.shuffle(RandomList)
            SummedData = [0] * PreviousMothersCount
            n = 0
            # Summing data based on the histogram heights and scars
            for j in range(len(PreviousHeights)):
                SummedData[n:n + PreviousHeights[j]] = RandomList[n:n + PreviousHeights[j]] + int(PreviousScars[j])
                n += PreviousHeights[j]
            Medians.append(int(median(SummedData)))
    FinalMedians[f'Experiment {Experiment}'] = Medians
    print(f'\n---------------------------------- Finished experiment {Experiment} ----------------------------------')

# Cleanup variables no longer needed
del PreviousMothersCount
del CombinedHistograms
del PreviousScars
del PreviousHeights
del CutMedians
del SummedData
del RandomList
del Experiment
del Medians
del i
del j
del m
del n

#%%%########################################################################## Calcofluor Only

# Define colors for the plots
Colors = ['darkviolet', 'mediumseagreen']

# Plotting scars data for each experiment separately
for n in range(1, NumberOfExperiments+1):
    # Create a new figure for each experiment
    plt.figure('Scars Experiment {}'.format(n))
    # Plot histogram of final scars for each experiment
    plt.hist(CalcoFluor['No. {}'.format(n)]['Scars_finales'], bins=int(max(CalcoFluor['No. {}'.format(n)]['Scars_finales'])-min(CalcoFluor['No. {}'.format(n)]['Scars_finales'])), label='Scars', edgecolor='black', color=Colors[n - 1], linewidth=1.2, align='left')
    plt.ylabel('Count', size=14)
    plt.xlabel('Generations', size=14)
    plt.title('Experiment {}'.format(n), size=14)
    plt.legend()
    plt.savefig(Path+'/Imagenes/Cicatrices Experimento {}.svg'.format(n), dpi=600)

# Plotting scars data for all experiments in a single figure with overlapping histograms
Colors = ['darkviolet', 'mediumseagreen']
plt.figure('Scars Experiment Comparison')
for n in range(1, NumberOfExperiments+1):
    # Plot overlapping histograms for scars data across experiments
    plt.hist(CalcoFluor['No. {}'.format(n)]['Scars_finales'], bins=int(max(CalcoFluor['No. {}'.format(n)]['Scars_finales'])-min(CalcoFluor['No. {}'.format(n)]['Scars_finales'])), label='Scars - Experiment {}'.format(n), edgecolor='black', color=Colors[n - 1], linewidth=1.2, align='left', alpha=0.5)
    plt.ylabel('Count', size=14)
    plt.xlabel('Generations', size=14)
    plt.legend()
# Save the figure
plt.savefig(Path+'/Imagenes/Cicatrices Experimentos Comparados superpuestos.svg', dpi=600)

# Combining data from all experiments for a single histogram
Colors = ['darkviolet', 'mediumseagreen']
CombinedData = []
Labels = []
Bins = []

plt.figure('Combined Scars Experiment Comparison')
for n in range(1, NumberOfExperiments+1):
    # Append data for each experiment to the combined data list
    CombinedData.append(list(CalcoFluor['No. {}'.format(n)]['Scars_finales']))
    # Calculate the number of bins based on the range of scars data
    Bins.append(int(max(CalcoFluor['No. {}'.format(n)]['Scars_finales'])-min(CalcoFluor['No. {}'.format(n)]['Scars_finales'])))
    Labels.append('Scars Experiment {}'.format(n))
# Plot histogram for the combined data
plt.hist(CombinedData, bins=max(Bins), label=Labels, edgecolor='black', color=Colors, linewidth=1.2, align='left')
plt.ylabel('Count', size=14)
plt.xlabel('Generations', size=14)
plt.legend()
plt.savefig(Path+'/Imagenes/Cicatrices Experimentos Comparados.svg', dpi=600)

# Clean up temporary variables
del Colors
del Labels
del CombinedData
del Bins
del n

#%%%########################################################################## Daughters Alone

Colors = ['darkviolet', 'mediumseagreen']
for n in range(1, NumberOfExperiments + 1):
    plt.figure(f'Generations Experiment {n}')
    plt.hist(Experiments[f'No. {n}']['Generations'], bins=int(max(Experiments[f'No. {n}']['Generations']) - min(Experiments[f'No. {n}']['Generations'])), label='Generations', edgecolor='black', color=Colors[n - 1], linewidth=1.2, align='left')
    plt.ylabel('Count', size=14)
    plt.xlabel('Generations', size=14)
    plt.title(f'Experiment {n}', size=14)
    plt.legend()
    plt.savefig(Path+'/Imagenes/Cicatrices Experimento {}.svg'.format(n), dpi=600)

Colors = ['darkviolet', 'mediumseagreen']
plt.figure('Generations Experiment')
for n in range(1, NumberOfExperiments + 1):
    plt.hist(Experiments[f'No. {n}']['Generations'], bins=int(max(Experiments[f'No. {n}']['Generations']) - min(Experiments[f'No. {n}']['Generations'])), label=f'Generations - Experiment {n}', edgecolor='black', color=Colors[n - 1], linewidth=1.2, align='left', alpha=0.5)
    plt.ylabel('Count', size=14)
    plt.xlabel('Generations', size=14)
    plt.legend()
plt.savefig(Path+'/Imagenes/Cicatrices Experimentos Comparados superpuestos.svg', dpi=600)

Colors = ['darkviolet', 'mediumseagreen']
Combined = []
Labels = []
Bins = []
plt.figure('Generations Experiment Compared')
for n in range(1, NumberOfExperiments + 1):
    Combined.append(list(Experiments[f'No. {n}']['Generations']))
    Bins.append(int(max(Experiments[f'No. {n}']['Generations']) - min(Experiments[f'No. {n}']['Generations'])))
    Labels.append(f'Generations Experiment {n}')
plt.hist(Combined, bins=max(Bins), label=Labels, edgecolor='black', color=Colors, linewidth=1.2, align='left')
plt.ylabel('Count', size=14)
plt.xlabel('Generations', size=14)
plt.legend()
plt.savefig(Path+'/Imagenes/Cicatrices Experimentos Comparados.svg', dpi=600)

# Clean up
del Colors
del Labels
del Combined
del Bins
del n

#%%%########################################################################## Calcofluor and Daughters

# Colors for the histograms
colors = ['darkviolet', 'mediumseagreen']

# Iterate through each experiment to create individual combined histograms
for n in range(1, NumberOfExperiments+1):
    plt.figure('Generations Experiment {}'.format(n))
    # Histogram for Scars (CalcoFluor)
    plt.hist(CalcoFluor['No. {}'.format(n)]['Scars_finales'], bins=int(max(CalcoFluor['No. {}'.format(n)]['Scars_finales'])-min(CalcoFluor['No. {}'.format(n)]['Scars_finales'])), label='Scars', edgecolor='black', color=colors[1], linewidth=1.2, align='left')
    # Histogram for Generations (Daughters)
    plt.hist(Experiments['No. {}'.format(n)]['Generations'], bins=int(max(Experiments['No. {}'.format(n)]['Generations'])-min(Experiments['No. {}'.format(n)]['Generations'])), label='Generations', edgecolor='black', color=colors[0], linewidth=1.2, align='left')
    plt.ylabel('Count', size=14)
    plt.xlabel('Generations', size=14)
    plt.title('Experiment {}'.format(n), size=14)
    plt.legend()
plt.savefig(Path+'/Imagenes/Cicatrices Experimentos Comparados superpuestos.svg', dpi=600)

# Colors for the combined histogram
colors = ['darkviolet', 'violet', 'mediumseagreen', 'green']

# Create a combined figure for all experiments
plt.figure('Generations Experiment')

for n in range(1, NumberOfExperiments+1):
    # Histogram for Generations with transparency (alpha)
    plt.hist(Experiments['No. {}'.format(n)]['Generations'], bins=int(max(Experiments['No. {}'.format(n)]['Generations'])-min(Experiments['No. {}'.format(n)]['Generations'])), label='Generations - Experiment {}'.format(n), edgecolor='black', color=colors[n + 1], linewidth=1.2, align='left', alpha=0.5)
    # Histogram for Scars with transparency (alpha)
    plt.hist(CalcoFluor['No. {}'.format(n)]['Scars_finales'], bins=int(max(CalcoFluor['No. {}'.format(n)]['Scars_finales'])-min(CalcoFluor['No. {}'.format(n)]['Scars_finales'])), label='Scars - Experiment {}'.format(n), edgecolor='black', color=colors[n - 1], linewidth=1.2, align='left', alpha=0.5)
    plt.ylabel('Count', size=14)
    plt.xlabel('Generations', size=14)
    plt.legend()

plt.savefig(Path+'/Imagenes/Cicatrices Experimentos Comparados.svg', dpi=600)

# Clean up temporary variables
del colors
del n

#%%%########################################################################## Comparison of Medians

# Define colors for the plots
colors = ['royalblue', 'green']

# Iterate through each experiment key in the Experiments dictionary
for n in list(Experiments.keys()):
    # Create a new figure for each experiment
    plt.figure(n)
    # Plot histogram of final medians for each experiment
    plt.hist(FinalMedians['Experiment '+n], bins=np.arange(13, 20, 1), edgecolor='black', linewidth=1.2, align='left', color=(255 / 255, 42 / 255, 42 / 255), label='Total Iterations\n{} - ☻'.format(TotalIterations*IterationsPerDistribution))
    plt.xlabel('Medians', size=14)
    plt.ylabel('Count', size=14)
    # Add a vertical line for the median of the generations in the current experiment
    plt.vlines(median(Experiments[n]['Generations']), 0, max(np.histogram(FinalMedians['Experiment {}'.format(n)])[0]), linewidth=5, color=(64 / 255, 66 / 255, 63 / 255), label='Measurements')
    plt.legend()
    plt.title('Median Shift with Experiment {}'.format(n), size=14)
    plt.savefig(Path+'/Imagenes/Corrimiento de medianas con Experimento {}.svg'.format(n), dpi=600)

# Clean up temporary variables
del IterationsPerDistribution
del NumberOfExperiments
del TotalIterations
del n
