import numpy as np
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

def EscapeFunction(NumberOfDaughters, EscapeFunction, Title, NoEscapeLabel, EscapeLabel):
    Data = {}
    Data['EscapeFunction'] = EscapeFunction
    if isinstance(EscapeFunction[0], np.bool_):
        DaughtersWithEscape = NumberOfDaughters[EscapeFunction]
        Data['DaughtersWithEscape'] = DaughtersWithEscape
    else:
        EscapePositions = np.array(NumberOfDaughters) < np.array(EscapeFunction)
        DaughtersWithEscape = np.array(NumberOfDaughters)[EscapePositions]
        Data['DaughtersWithEscape'] = DaughtersWithEscape
    EscapePlot, ax = plt.subplots(figsize=(18, 12))
    Heights, Bins, _ = ax.hist(NumberOfDaughters, bins=max(NumberOfDaughters), edgecolor='black', linewidth=2, density=False, color='lightseagreen', label=NoEscapeLabel, align='left')
    ax.hist(DaughtersWithEscape, bins=Bins, edgecolor='black', linewidth=2, density=False, color='darkviolet', label=EscapeLabel, align='left')
    ax.set_title(Title, pad=8, loc='center', color='darkviolet', font={'family': 'cursive', 'weight': 'bold', 'size': 52})
    ax.set_xlabel('Generation', labelpad=3.2, loc='center', size=32, color='black')
    ax.set_ylabel('Count', labelpad=3.2, loc='center', size=32, color='black')
    ax.vlines(np.median(NumberOfDaughters), 0, max(Heights) + max(Heights) * 15 / 100, linewidth=5, linestyle='dashed', color='lightseagreen')
    ax.vlines(np.median(DaughtersWithEscape), 0, max(Heights) + max(Heights) * 15 / 100, linewidth=5, linestyle='dashed', color='darkviolet')
    ax.set_xlim([-1, 71])
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

MaxGenerationsToUse = 100

Data = InitialPopulationTimes(MaxGenerations=MaxGenerationsToUse)

RandomEscapeData = EscapeFunction(Title='Random escape', NoEscapeLabel='No escape', EscapeLabel='Age-independant escape', NumberOfDaughters=Data['NumberOfDaughters'], EscapeFunction=RandomEscape(Data['NumberOfDaughters'], 0, 100))

NonRandomEscapeData = EscapeFunction(Title='Non-Random escape', NoEscapeLabel='No escape', EscapeLabel='Age-independant escape', NumberOfDaughters=Data['NumberOfDaughters'], EscapeFunction=LinearEscape(Data['NumberOfDaughters'], 0, 100))

# np.save(r'D:\JøFrå\Estudio\Facultad\Fisica\Science Bitch\Datos\Simulacion', Data)

# np.save(r'D:\JøFrå\Estudio\Facultad\Fisica\Science Bitch\Datos\Simulacion Random', RandomEscapeData)

# np.save(r'D:\JøFrå\Estudio\Facultad\Fisica\Science Bitch\Datos\Simulacion no Random', NonRandomEscapeData)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% RLS

MothersWithoutLoss = np.isfinite(Data['TrimmedLifeTimes'])
MothersWithoutLossRLS = np.sum(MothersWithoutLoss, 0)

RLSWithoutLoss = MothersWithoutLossRLS/MothersWithoutLossRLS.max()

MothersWithLoss = np.isfinite(Data['TrimmedLifeTimes'])
MothersWithLossRLS = np.sum(MothersWithLoss[RandomEscapeData['EscapeFunction']], 0)

RLSWithLoss = MothersWithLossRLS/MothersWithLossRLS.max()

RLSGraphIndependant, ax = plt.subplots(figsize=(18, 12))
ax.set_title('Comparison of RLS', pad=8, loc='center', color='darkviolet', font={'family': 'cursive', 'weight': 'bold', 'size': 52})
ax.scatter(np.arange(0, MaxGenerationsToUse, 1), RLSWithoutLoss, label='No escape', color='lightseagreen', s = 200)
ax.scatter(np.arange(0, MaxGenerationsToUse, 1), RLSWithLoss, label='Age-independant escape', color='darkviolet', s = 200, marker = 'X')
ax.set_xlim([0, 80])
ax.set_xlabel('Generation', labelpad=3.2, loc='center', size=32, color='black')
ax.set_ylabel('Fraction Viable', labelpad=3.2, loc='center', size=32, color='black')
ax.tick_params(axis='both', which='major', labelsize=25)
ax.legend(fontsize=25)

#%%

MothersWithoutLoss = np.isfinite(Data['TrimmedLifeTimes'])
MothersWithoutLossRLS = np.sum(MothersWithoutLoss, axis=0)

RLSWithoutLoss = MothersWithoutLossRLS/MothersWithoutLossRLS.max()

MothersWithLoss = np.isfinite(Data['TrimmedLifeTimes'])
EscapedPositions = Data['NumberOfDaughters']<NonRandomEscapeData['EscapeFunction']
MothersWithLossRLS = np.sum(MothersWithLoss[EscapedPositions], axis=0)

RLSWithLoss = MothersWithLossRLS/MothersWithLossRLS.max()

RLSGraphDependent, ax = plt.subplots(figsize=(18, 12))
ax.set_title('Comparison of RLS', pad=8, loc='center', color='darkviolet', font={'family': 'cursive', 'weight': 'bold', 'size': 52})
ax.scatter(np.arange(0, MaxGenerationsToUse, 1), RLSWithoutLoss, label='No escape', color='lightseagreen', s = 200)
ax.scatter(np.arange(0, MaxGenerationsToUse, 1), RLSWithLoss, label='Age-dependent escape', color='darkviolet', s = 200, marker = 'X')
ax.set_xlim([0, 80])
ax.set_xlabel('Generation', labelpad=3.2, loc='center', size=32, color='black')
ax.set_ylabel('Fraction Viable', labelpad=3.2, loc='center', size=32, color='black')
ax.tick_params(axis='both', which='major', labelsize=25)
ax.legend(fontsize=25)