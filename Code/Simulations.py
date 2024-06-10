import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy import random as rnd
from scipy.special import gamma
from numpy import matlib as mtlb

Path = input('Please, enter the path name where the data is located\nPath=')

"""
This code consists of three cells:
    1°)Generates the necessary general functions.
    2°)Creates new simulated data by applying the simulation.
    3°)Reproduces the figures shown in the paper by plotting the simulated data.
    To use the last cell, you need to download the file named "6A 6B - Simulated Data".
"""

#%%% GENERAL FUNTCIONS %%%#

#%to do the RLS curve
def Accumulated(DataSorted): 
    Step = []
    n=0
    for i in range(0,len(DataSorted)):
        if DataSorted[i] == DataSorted[i-1]:
            n+=1
            Step.append(i+n)
        else:
            Step.append(i+n)
    return(np.flip(Step))

#%Population
# The following function generates an amount of generations for "NumberOfCells" mother cells, which are given by a Weibull distribution.
def InitialPopulationTimes(NumberOfCells=6000, MaxGenerations=100, Exponent=-0.4, AverageBirthTime=90, BirthTimeStdDev=2, AverageGenerations=25, GenerationsStdDev=5, WeibullK=2, Asymmetry=True, InvertExponential=True):
    Data = {}
    Data['Metadata'] = {'NumberOfCells': NumberOfCells, 'MaxGenerations': MaxGenerations, 'Exponent': Exponent, 'AverageBirthTime': AverageBirthTime, 'BirthTimeStdDev': BirthTimeStdDev, 'AverageGenerations': AverageGenerations, 'GenerationsStdDev': GenerationsStdDev, 'WeibullK': WeibullK, 'Asymmetry': Asymmetry, 'InvertExponential': InvertExponential}
    if InvertExponential == True:
        NumberOfDaughters = np.flip(np.arange(1, MaxGenerations + 1))
    else:
        NumberOfDaughters = np.arange(1, MaxGenerations + 1)
    BirthTimeTrend = AverageBirthTime * NumberOfDaughters**Exponent  # Time variation between daughters across generations.
    if InvertExponential == True:
        BirthTimeTrend = BirthTimeTrend - BirthTimeTrend[0]  # Time differences between daughters across generations compared to the tabulated value (from lower to higher - inverted).
    else:
        BirthTimeTrend = BirthTimeTrend - BirthTimeTrend[-1]  # Time differences between daughters across generations compared to the tabulated value (from higher to lower).
    BirthTimes = rnd.normal(loc=AverageBirthTime, scale=BirthTimeStdDev, size=(NumberOfCells, MaxGenerations)) + mtlb.repmat(BirthTimeTrend, NumberOfCells, 1)
    WeibullParameters = AverageGenerations / gamma(1 + 1 / WeibullK)
    BirthsPerMother = np.full((NumberOfCells, MaxGenerations), np.nan)
    if Asymmetry == True:
        DaughtersPerMother = (WeibullParameters * rnd.weibull(WeibullK, size=(1, NumberOfCells)))[0].round(0).astype(int)
    else:
        DaughtersPerMother = abs(rnd.normal(loc=AverageGenerations, scale=GenerationsStdDev, size=NumberOfCells).round(0).astype(int))
    for Row in range(NumberOfCells):
        BirthsPerMother[Row] = np.array(list(BirthTimes[Row][:DaughtersPerMother[Row]]) + [np.nan] * (MaxGenerations - DaughtersPerMother[Row]))
    Data['BirthTimes'] = BirthTimes
    Data['BirthTimeTrend'] = BirthTimeTrend
    Data['NumberOfDaughters'] = DaughtersPerMother
    Data['TrimmedLifeTimes'] = BirthsPerMother
    return(Data)

#% Escape aplicator
def EscapeFunction(NumberOfDaughters, GenerationTimes, EscapeFunction, TitleAndSave, NoEscapeLabel, EscapeLabel, NoEscapeColor, EscapeColor):
    if TitleAndSave != False:
        if not os.path.exists(Path+'\Figures'): 
            os.makedirs(Path+'\Figures')
    Data = {}
    Data['EscapeFunction'] = EscapeFunction
    RLSWithoutLoss = Accumulated(np.sort(NumberOfDaughters))
    RLSWithoutLoss = RLSWithoutLoss/max(RLSWithoutLoss)
    Data['RLSWithoutLoss'] = RLSWithoutLoss
    if isinstance(EscapeFunction[0], np.bool_):
        DaughtersWithEscape = NumberOfDaughters[EscapeFunction]
        Data['DaughtersWithEscape'] = DaughtersWithEscape
        RLSWithLoss = Accumulated(np.sort(DaughtersWithEscape))
        RLSWithLoss = RLSWithLoss/max(RLSWithLoss)
        Data['RLSWithLoss'] = RLSWithLoss
    else:
        EscapePositions = np.array(NumberOfDaughters)<np.array(EscapeFunction)
        DaughtersWithEscape = np.array(NumberOfDaughters)[EscapePositions]
        Data['DaughtersWithEscape'] = DaughtersWithEscape
        RLSWithLoss = Accumulated(np.sort(DaughtersWithEscape))
        RLSWithLoss = RLSWithLoss/max(RLSWithLoss)
        Data['RLSWithLoss'] = RLSWithLoss
    EscapePlot, ax = plt.subplots(figsize=(18, 12))
    Heights, Bins, _ = ax.hist(NumberOfDaughters, bins=max(NumberOfDaughters), edgecolor='black', linewidth=2, density=False, color=NoEscapeColor, label=NoEscapeLabel, align='left') #'lightseagreen'
    ax.hist(DaughtersWithEscape, bins=Bins, edgecolor='black', linewidth=2, density=False, color=EscapeColor, label=EscapeLabel, align='left') #'darkviolet'
    ax.set_xlabel('Generation', labelpad=3.2, loc='center', size=32, color='black')
    ax.set_ylabel('Count', labelpad=3.2, loc='center', size=32, color='black')
    ax.vlines(np.median(NumberOfDaughters), 0, max(Heights)+max(Heights)*15/100, linewidth=5, linestyle='dashed', color=NoEscapeColor)
    ax.vlines(np.median(DaughtersWithEscape), 0, max(Heights)+max(Heights)*15/100, linewidth=5, linestyle='dotted', color=EscapeColor)
    ax.set_xlim([-1, 71])
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.legend(fontsize=25)
    if TitleAndSave != False:
        ax.set_title(TitleAndSave+' Using new generated data', pad=8, loc='center', fontsize=52)
        plt.savefig(Path+'\Figures\(New simulated Data) '+TitleAndSave+'.png', dpi=600)
    else:
        ax.set_title('Using new generated data', pad=8, loc='center', fontsize=52)
    RLSGraphIndependant, ax = plt.subplots(figsize=(18, 12))
    ax.scatter(np.sort(NumberOfDaughters), RLSWithoutLoss, label=NoEscapeLabel, color=NoEscapeColor, s = 200)
    ax.scatter(np.sort(DaughtersWithEscape), RLSWithLoss, label=EscapeLabel, color=EscapeColor, s = 200, marker = 'X')
    ax.set_xlabel('Generation', labelpad=3.2, loc='center', size=32, color='black')
    ax.set_ylabel('Fraction Viable', labelpad=3.2, loc='center', size=32, color='black')
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.legend(fontsize=25)
    if TitleAndSave != False:
        ax.set_title('Using new generated data', pad=8, loc='center', fontsize=52)
        plt.savefig(Path+'\Figures\(RLS new simulated Data) '+TitleAndSave+'.png', dpi=600)
    else:
        ax.set_title('Using new generated data', pad=8, loc='center', fontsize=52)
    return(Data)

#% Diferents Escapes

def RandomEscape(NumberOfDaughters, Min, Max):
    LinearDeath1 = rnd.randint(Min, Max, len(NumberOfDaughters))
    LinearDeath2 = rnd.randint(Min, Max, len(NumberOfDaughters))
    RandomDeath = LinearDeath1<LinearDeath2
    return RandomDeath

def LinearEscape(NumberOfDaughters, Min, Max):
    LinearDeath = rnd.randint(Min, Max, len(NumberOfDaughters)) / 2
    return LinearDeath

#%% APPLICATION

Data = InitialPopulationTimes()
RandomEscapeData = EscapeFunction(NoEscapeLabel='No escape', EscapeLabel='Age-independant escape', NumberOfDaughters=Data['NumberOfDaughters'], GenerationTimes=Data['TrimmedLifeTimes'], EscapeFunction=RandomEscape(Data['NumberOfDaughters'], 0, 100), NoEscapeColor = (255/255, 160/255, 0/255), EscapeColor = (255/255, 0/255, 0/255), TitleAndSave = False)
NonRandomEscapeData = EscapeFunction(NoEscapeLabel='No escape', EscapeLabel='Age-dependant escape', NumberOfDaughters=Data['NumberOfDaughters'], GenerationTimes=Data['TrimmedLifeTimes'], EscapeFunction=LinearEscape(Data['NumberOfDaughters'], 0, 100), NoEscapeColor = (255/255, 160/255, 0/255), EscapeColor = (0/255, 0/255, 255/255), TitleAndSave = False)

#%SAVE DATA

if not os.path.exists(Path+r'\New Data'): 
    os.makedirs(Path+r'\New Data') 

# np.save(r'D:\JøFrå\Estudio\Facultad\Fisica\Datos\Simulacion', Data) # Dictionary saved as .npy containing all the information of the cell population created by the function InitialPopulationTimes.
# np.save(r'D:\JøFrå\Estudio\Facultad\Fisica\Datos\Simulacion Random', RandomEscapeData) # Dictionary saved as .npy containing all the information of the cell population with the escape "RandomEscapeData" applied.
# np.save(r'D:\JøFrå\Estudio\Facultad\Fisica\Datos\Simulacion no Random', NonRandomEscapeData) # Dictionary saved as .npy containing all the information of the cell population with the escape "RandomEscapeData" applied.
HistogramData = pd.DataFrame({"No escape": list(Data['NumberOfDaughters']), "Age-independant escape": list(RandomEscapeData['DaughtersWithEscape'])+[None]*(len(Data['NumberOfDaughters'])-len(RandomEscapeData['DaughtersWithEscape'])), "Age-dependant escape": list(NonRandomEscapeData['DaughtersWithEscape'])+[None]*(len(Data['NumberOfDaughters'])-len(NonRandomEscapeData['DaughtersWithEscape']))})
HistogramData.to_csv(Path + r'\New Data\6A 6B - Simulated Data.csv', index=False) # CSV with the histograms information.

del RandomEscapeData
del NonRandomEscapeData

#%%% DATA LOADING AND VISUALIZATION %%%#

Data = pd.read_csv(Path + r'\6A 6B - Simulated Data.csv', delimiter=',')

IndependantPlot, ax = plt.subplots(figsize=(18, 12))
Heights, Bins, _ = ax.hist(Data['No escape'], bins=max(Data['No escape']), edgecolor='black', linewidth=2, density=False, color=(255/255, 160/255, 0/255), label='No escape', align='left')
ax.hist(Data['Age-independant escape'], bins=Bins, edgecolor='black', linewidth=2, density=False, color=(255/255, 0/255, 0/255), label='Age-independant escape', align='left')
ax.set_xlabel('Generation', labelpad=3.2, loc='center', size=32, color='black')
ax.set_ylabel('Count', labelpad=3.2, loc='center', size=32, color='black')
ax.vlines(np.median(Data['No escape']), 0, max(Heights)+max(Heights)*15/100, linewidth=5, linestyle='dashed', color=(255/255, 160/255, 0/255))
ax.vlines(np.median(Data['Age-independant escape'].dropna()), 0, max(Heights)+max(Heights)*15/100, linewidth=5, linestyle='dotted', color=(255/255, 0/255, 0/255))
ax.set_xlim([-1, 71])
ax.tick_params(axis='both', which='major', labelsize=25)
plt.title('Using data from DATA folder', fontsize = 42, pad = 18, loc = 'center')
ax.legend(fontsize=25)

DependantPlot, ax = plt.subplots(figsize=(18, 12))
Heights, Bins, _ = ax.hist(Data['No escape'], bins=max(Data['No escape']), edgecolor='black', linewidth=2, density=False, color=(255/255, 160/255, 0/255), label='No escape', align='left')
ax.hist(Data['Age-dependant escape'], bins=Bins, edgecolor='black', linewidth=2, density=False, color=(0/255, 0/255, 255/255), label='Age-dependant escape', align='left')
ax.set_xlabel('Generation', labelpad=3.2, loc='center', size=32, color='black')
ax.set_ylabel('Count', labelpad=3.2, loc='center', size=32, color='black')
ax.vlines(np.median(Data['No escape']), 0, max(Heights)+max(Heights)*15/100, linewidth=5, linestyle='dashed', color=(255/255, 160/255, 0/255))
ax.vlines(np.median(Data['Age-dependant escape'].dropna()), 0, max(Heights)+max(Heights)*15/100, linewidth=5, linestyle='dashed', color=(0/255, 0/255, 255/255))
ax.set_xlim([-1, 71])
ax.tick_params(axis='both', which='major', labelsize=25)
plt.title('Using data from DATA folder', fontsize = 42, pad = 18, loc = 'center')
ax.legend(fontsize=25)

# Create RLSs curve since histogram data
DataSortedNoEscape = np.sort(Data['No escape'].dropna())
DataSortedDependantEscape = np.sort(Data['Age-dependant escape'].dropna())
DataSortedIndependantEscape = np.sort(Data['Age-independant escape'].dropna())

RLSNoEscape = Accumulated(DataSortedNoEscape)
RLSDependant = Accumulated(DataSortedDependantEscape)
RLSIndependant = Accumulated(DataSortedIndependantEscape)

RLSs = {'No escape': RLSNoEscape/max(RLSNoEscape), 'Age-independant escape': RLSIndependant/max(RLSIndependant), 'Age-dependant escape': RLSDependant/max(RLSDependant)} 

RLSGraphIndependant, ax = plt.subplots(figsize=(18, 12))
ax.scatter(DataSortedNoEscape, RLSs['No escape'], label='No escape', color=(255/255, 160/255, 0/255), s = 200)
ax.scatter(DataSortedIndependantEscape, RLSs['Age-independant escape'], label='Age-independant escape', color=(255/255, 0/255, 0/255), s = 200, marker = 'X')
ax.set_xlabel('Generation', labelpad=3.2, loc='center', size=32, color='black')
ax.set_ylabel('Fraction Viable', labelpad=3.2, loc='center', size=32, color='black')
ax.tick_params(axis='both', which='major', labelsize=25)
plt.title('Using data from DATA folder', fontsize = 42, pad = 18, loc = 'center')
ax.legend(fontsize=25)

RLSGraphDependant, ax = plt.subplots(figsize=(18, 12))
ax.scatter(DataSortedNoEscape, RLSs['No escape'], label='No escape', color=(255/255, 160/255, 0/255), s = 200)
ax.scatter(DataSortedDependantEscape, RLSs['Age-dependant escape'], label='Age-dependant escape', color=(0/255, 0/255, 255/255), s = 200, marker = 'X')
ax.set_xlabel('Generation', labelpad=3.2, loc='center', size=32, color='black')
ax.set_ylabel('Fraction Viable', labelpad=3.2, loc='center', size=32, color='black')
ax.tick_params(axis='both', which='major', labelsize=25)
plt.title('Using data from DATA folder', fontsize = 42, pad = 18, loc = 'center')
ax.legend(fontsize=25)

# Cleanup variables no longer needed
del ax
del Bins
del RLSs
del Heights
del RLSNoEscape
del RLSDependant
del DependantPlot
del RLSIndependant
del IndependantPlot
del RLSGraphDependant
del DataSortedNoEscape
del RLSGraphIndependant
del DataSortedDependantEscape
del DataSortedIndependantEscape

