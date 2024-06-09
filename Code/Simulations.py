import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy import random as rnd
from scipy.special import gamma
from numpy import matlib as mtlb

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% POPULATION FUNCTION

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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% APPLICATION OF ESCAPE FUNCTION

def EscapeFunction(NumberOfDaughters, GenerationTimes, EscapeFunction, Title, NoEscapeLabel, EscapeLabel, NoEscapeColor, EscapeColor):
    Data = {}
    Data['EscapeFunction'] = EscapeFunction
    if isinstance(EscapeFunction[0], np.bool_):
        DaughtersWithEscape = NumberOfDaughters[EscapeFunction]
        Data['DaughtersWithEscape'] = DaughtersWithEscape
        MothersWithoutLoss = np.isfinite(GenerationTimes)
        MothersWithoutLossRLS = np.sum(MothersWithoutLoss, 0)
        RLSWithoutLoss = MothersWithoutLossRLS/MothersWithoutLossRLS.max()
        Data['RLSWithoutLoss'] = RLSWithoutLoss
        MothersWithLoss = np.isfinite(GenerationTimes)
        MothersWithLossRLS = np.sum(MothersWithLoss[EscapeFunction], 0)
        RLSWithLoss = MothersWithLossRLS/MothersWithLossRLS.max()
        Data['RLSWithLoss'] = RLSWithLoss
    else:
        EscapePositions = np.array(NumberOfDaughters)<np.array(EscapeFunction)
        DaughtersWithEscape = np.array(NumberOfDaughters)[EscapePositions]
        Data['DaughtersWithEscape'] = DaughtersWithEscape
        MothersWithoutLoss = np.isfinite(GenerationTimes)
        MothersWithoutLossRLS = np.sum(MothersWithoutLoss, axis=0)
        RLSWithoutLoss = MothersWithoutLossRLS/MothersWithoutLossRLS.max()
        Data['RLSWithoutLoss'] = RLSWithoutLoss
        MothersWithLoss = np.isfinite(GenerationTimes)
        MothersWithLossRLS = np.sum(MothersWithLoss[EscapePositions], axis=0)
        RLSWithLoss = MothersWithLossRLS/MothersWithLossRLS.max()
        Data['RLSWithLoss'] = RLSWithLoss
    EscapePlot, ax = plt.subplots(figsize=(18, 12))
    Heights, Bins, _ = ax.hist(NumberOfDaughters, bins=max(NumberOfDaughters), edgecolor='black', linewidth=2, density=False, color=NoEscapeColor, label=NoEscapeLabel, align='left') #'lightseagreen'
    ax.hist(DaughtersWithEscape, bins=Bins, edgecolor='black', linewidth=2, density=False, color=EscapeColor, label=EscapeLabel, align='left') #'darkviolet'
    ax.set_xlabel('Generation', labelpad=3.2, loc='center', size=32, color='black')
    ax.set_ylabel('Count', labelpad=3.2, loc='center', size=32, color='black')
    ax.vlines(np.median(NumberOfDaughters), 0, max(Heights) + max(Heights) * 15 / 100, linewidth=5, linestyle='dashed', color=NoEscapeLabel)
    ax.vlines(np.median(DaughtersWithEscape), 0, max(Heights) + max(Heights) * 15 / 100, linewidth=5, linestyle='dashed', color=EscapeLabel)
    ax.set_xlim([-1, 71])
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.legend(fontsize=25)
    RLSGraphIndependant, ax = plt.subplots(figsize=(18, 12))
    ax.scatter(np.arange(0, len(RLSWithoutLoss), 1), RLSWithoutLoss, label=NoEscapeLabel, color=NoEscapeColor, s = 200)
    ax.scatter(np.arange(0, len(RLSWithLoss), 1), RLSWithLoss, label=EscapeLabel, color=EscapeColor, s = 200, marker = 'X')
    ax.set_xlabel('Generation', labelpad=3.2, loc='center', size=32, color='black')
    ax.set_ylabel('Fraction Viable', labelpad=3.2, loc='center', size=32, color='black')
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.legend(fontsize=25)
    return(Data)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ESCAPE FUNCTION

def RandomEscape(NumberOfDaughters, Min, Max):
    LinearDeath1 = rnd.randint(Min, Max, len(NumberOfDaughters))
    LinearDeath2 = rnd.randint(Min, Max, len(NumberOfDaughters))
    RandomDeath = LinearDeath1<LinearDeath2
    return RandomDeath

def LinearEscape(NumberOfDaughters, Min, Max):
    LinearDeath = rnd.randint(Min, Max, len(NumberOfDaughters)) / 2
    return LinearDeath

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% APPLICATIONS

Data = InitialPopulationTimes()

RandomEscapeData = EscapeFunction(Title='Random escape', NoEscapeLabel='No escape', EscapeLabel='Age-independant escape', NumberOfDaughters=Data['NumberOfDaughters'], GenerationTimes=Data['TrimmedLifeTimes'], EscapeFunction=RandomEscape(Data['NumberOfDaughters'], 0, 100), NoEscapeColor = (255/255, 160/255, 0/255), EscapeColor = (255/255, 0/255, 0/255))

NonRandomEscapeData = EscapeFunction(Title='Non-Random escape', NoEscapeLabel='No escape', EscapeLabel='Age-dependant escape', NumberOfDaughters=Data['NumberOfDaughters'], GenerationTimes=Data['TrimmedLifeTimes'], EscapeFunction=LinearEscape(Data['NumberOfDaughters'], 0, 100), NoEscapeColor = (255/255, 160/255, 0/255), EscapeColor = (0/255, 0/255, 255/255))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Data saved

# np.save(r'D:\JøFrå\Estudio\Facultad\Fisica\Datos\Simulacion', Data) # Dictionary saved as .npy containing all the information of the cell population created by the function InitialPopulationTimes.

# np.save(r'D:\JøFrå\Estudio\Facultad\Fisica\Datos\Simulacion Random', RandomEscapeData) # Dictionary saved as .npy containing all the information of the cell population with the escape "RandomEscapeData" applied.

# np.save(r'D:\JøFrå\Estudio\Facultad\Fisica\Datos\Simulacion no Random', NonRandomEscapeData) # Dictionary saved as .npy containing all the information of the cell population with the escape "RandomEscapeData" applied.

HistogramData = pd.DataFrame({"No escape": list(Data['NumberOfDaughters']), "Age-independant escape": list(RandomEscapeData['DaughtersWithEscape'])+[None]*(len(Data['NumberOfDaughters'])-len(RandomEscapeData['DaughtersWithEscape'])), "Age-dependant escape": list(NonRandomEscapeData['DaughtersWithEscape'])+[None]*(len(Data['NumberOfDaughters'])-len(NonRandomEscapeData['DaughtersWithEscape']))})
HistogramData.to_csv(r'D:\JøFrå\Estudio\Facultad\Fisica\Datos\Histogram Data.csv', index=False) # CSV with the histograms information.

RLSData = pd.DataFrame({"No escape": list(RandomEscapeData['RLSWithoutLoss']), "Age-independant escape": list(RandomEscapeData['RLSWithLoss']), "Age-dependant escape": list(NonRandomEscapeData['RLSWithLoss'])})
RLSData.to_csv(r'D:\JøFrå\Estudio\Facultad\Fisica\Datos\RLS Data.csv', index=False) # CSV with the RLSs information.


