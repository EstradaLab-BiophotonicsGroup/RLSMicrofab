import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statistics import median

Path = r'D:/JøFrå/Estudio/Facultad/Fisica'

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

#%%########################################################################### IMAGES

#%%%########################################################################## Calcofluor and Daughters

plt.figure('Generations All Together')
# Histogram for Scars (CalcoFluor)
plt.hist(CalcoFluor['All Together']['Scars_finales'], bins=int(max(CalcoFluor['All Together']['Scars_finales'])-min(CalcoFluor['All Together']['Scars_finales'])), label='$t=0$, initial population', edgecolor='black', color=(192/255, 192/255, 192/255), linewidth=1.2, align='left')
# Histogram for Generations (Daughters)
plt.hist(Experiments['All Together']['Generations'], bins=int(max(Experiments['All Together']['Generations'])-min(Experiments['All Together']['Generations'])), label='$t>0$, complete RLS experiment', edgecolor='black', color=(255/255, 177/255, 77/255), linewidth=1.2, align='left')
plt.vlines(median(Experiments['All Together']['Generations']), 0, max(np.histogram(Experiments['All Together']['Generations'])[0])-max(np.histogram(Experiments['All Together']['Generations'])[0])*5/100, linewidth=2, color=(228/255, 163/255, 88/255))
plt.vlines(median(CalcoFluor['All Together']['Scars_finales']), 0, max(np.histogram(Experiments['All Together']['Generations'])[0])-max(np.histogram(Experiments['All Together']['Generations'])[0])*5/100, linewidth=2, color=(111/255, 111/255, 111/255))
plt.ylabel('Count', size=14)
plt.xlabel('Generations', size=14)
plt.legend()
plt.savefig(Path+'/Imagenes/Datos juntos vs cicatrices juntas.png', dpi=600)

#%%%########################################################################## Comparison of Medians

plt.figure('Median Shift')
# Plot histogram of final medians for each experiment
plt.hist(FinalMedians['Experiment All Together'], bins=np.arange(13, 20, 1), edgecolor='black', linewidth=1.2, rwidth = 0.5, align='left', color=(0/255, 10/255, 10/255))
# Add a vertical line for the median of the generations in the current experiment
plt.vlines(median(Experiments['All Together']['Generations']), 0, max(np.histogram(FinalMedians['Experiment All Together'])[0])+max(np.histogram(FinalMedians['Experiment All Together'])[0])*5/100, linewidth=2, color=(255/255, 177/255, 77/255))
plt.xlabel('Medians', size=14)
plt.ylabel('Count', size=14)
plt.savefig(Path+'/Imagenes/Corrimiento de medianas juntas.png', dpi=600)
