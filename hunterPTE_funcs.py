import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import palettable

from HLAPredCache import *
from custom_legends import colorLegend
from seqdistance import distance_rect
from seqdistance import nbmetrics
from seqdistance import seqs2mat
from seqdistance import distance_df
from seqtools import *

"""Site vectors (sitevec), xticks (t) and xtick labels (tl)"""
pctt = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
pcttl = (pctt * 100).astype(int)

def isforbiddenPep(pep):
    """IAVI criteria: Peptides with Q or N at N-terminus or P or C at C-terminus cannot be manufactured.

    Peptides with Q at N-terminus or GPEDQNTSC at C-terminus cannot be manufactured.
    It is both a manufacturing and solubility issue, along with an empirical issue that peptides with these AAs are never epitopes (underlying mechanism may be the same)
    Frahm N, Korber B, Adams C, Szinger J, Addo M, Feeney M, et al. Consistent cytotoxic-T-lymphocyte targeting of immunodominant regions in human immunodeficiency virus across multiple ethnicities. J Virol 2004;78:2187-200. doi:10.1128/JVI.78.5.2187.
    Draenert R, Brander C, Xu G, Altfeld M. Impact of intrapeptide epitope location on CD8 T cell recognition: implications for design of overlapping peptide panels. Aids 2004. doi:10.1097/01.aids.0000111400.02002.db."""
    if pep is None:
        return True
    else:
        return pep[0] in 'QN' or pep[-1] in 'PC'
    #return pep[0] in 'Q' or pep[-1] in 'GPEDQNTSC'

def fixforbiddenPep(seq, L, starti):
    _, pep = grabKmer(seq, starti, L)
    origPep = pep
    if not pep is None and pep[0] in 'QN':
        _, newPep = grabKmer(seq, starti-1, L+1)
        if not newPep is None and not newPep[0] in 'QN':
            starti -= 1
            L += 1
        else:
            starti += 1
            L-= 1
    _, pep = grabKmer(seq, starti, L)
    if not pep is None and pep[-1] in 'PC':
        _, newPep = grabKmer(seq, starti, L+1)
        if not newPep is None and not newPep[-1] in 'PC':
            L += 1
        else:
            L -= 1
    _, pep = grabKmer(seq, starti, L)
    if isforbiddenPep(pep):
        print('Applying a double-fix to peptide %s' % origPep)
        return fixforbiddenPep(seq, starti, L)
    else:
        return pep

def removeAxes(axh, remove=['right', 'top']):
    if 'right' in remove:
        axh.spines['right'].set_visible(False)
        axh.yaxis.set_ticks_position('left')
    if 'top' in remove:
        axh.spines['top'].set_visible(False)
        axh.xaxis.set_ticks_position('bottom')

def matchedPTIDs(r, non=True, mmTolerance=1):
    ptids = []
    for ptid in seqDf.index:
        ptidPep = seqDf['seq'].loc[ptid][r['starti']:r['starti']+len(r['peptide'])]
        hdist = hamming_distance(ptidPep, r['peptide'])
        if hdist <= mmTolerance:
            ptids.append(ptid)
    if non:
        ptids = [p for p in seqDf.index if not p in ptids]
    return '/'.join(sorted(ptids))

def predictedHLAs(r, bindingThreshold=np.log(500), twoDigit=True):
    hlas = []
    for mer in getMers(r['peptide'], nmer=[9]):
        for h in uHLAs:
            if ba[(h, mer)] < bindingThreshold:
                hlas.append(h)
    if twoDigit:
        hlas = [h[:4] for h in hlas]
    hlas = list({h.replace('_', '*') for h in hlas})
    return '/'.join(sorted(hlas))

def predictedPTIDs(r, non=True, mmTolerance=1, bindingThreshold=np.log(500)):
    hlas = []
    for mer in getMers(r['peptide'], nmer=[9]):
        for h in uHLAs:
            if ba[(h, mer)] < bindingThreshold:
                hlas.append(h)
    hlas = set(hlas)
    ptids = []
    for ptid in seqDf.index:
        ptidPep = seqDf['seq'].loc[ptid][r['starti']:r['starti']+len(r['peptide'])]
        hdist = hamming_distance(ptidPep, r['peptide'])
        expressed = np.any([h in hlas for h in hlaDf['hlas'].loc[ptid]])
        if hdist <= mmTolerance and expressed:
            ptids.append(ptid)
    if non:
        ptids = [p for p in seqDf.index if not p in ptids]
    return '/'.join(sorted(ptids))

def fractionConsensus(row):
    region = (row['starti'], row['starti'] + len(row['peptide']))
    algn = sliceAlign(seqDf.seq, region=region)
    return (algn == row['ConsensusPeptide']).mean()

def alignKnownEpitopes(df, conSeq):
    findPos = getStartPosMapper(conSeq, _substMat)
    df['sitei'] = df.Epitope.map(findPos)

    sliceSeqFunc = lambda row: conSeq[row['sitei']:row['sitei']+len(row['Epitope'])]
    df['conEpitope'] = df.apply(sliceSeqFunc, axis=1)

    hdistFunc = lambda row: hamming_distance(row['Epitope'], row['conEpitope'])
    df['hdist'] = df.apply(hdistFunc, axis=1)
    df.drop_duplicates(subset=['conEpitope'])
    return df

def coverageOptimizer(pa, k, mmTolerance=1, epitopesOnly=False, bindingThreshold=np.log(500)):
    """Select the k kmers that provide the best coverage overall.
    Changes are made to the pa object directly.
    Considers coverage at each position independently.
    Optionally, only considers coverage of kmers that are predicted to bind HLA.


    Algorithm:
    Iterively select the peptide across all positions that provides maximal coverage.

    Not optimal since always picking the next best peptide for coverage is not as good as
    picking N best peptides with N previously known. It is optimal for mmTolerance=1 though.

    ISSUE: this algorithm treats each position completely independently. One consequence of
    this is that often when a variant is selected for one position the same selection is made
    in the adjacent position at the next step, creating blocks of selected peptides that
    are all very similar. If epitopes were exactly 9mers and start positions were
    indendent then this would be desirable, but because they are not it seems like the wrong
    solution.

    Issue example:
    out = coverageOptimizer(pa, 300, mmTolerance=0, epitopesOnly=False)
    out.loc[out.pos.isin(out.pos.loc[out.pos_count==2])].sort(columns=['pos'])

    Parameters
    ----------
    pa : PeptidAllocation object
    k : int
        Number of peptides to select.
    mmThreshold : int
        Number of mismatches tolerated by coverage calculation.
    epitopeThreshold : None or float
        If not None then the log-IC50 binding threshold defining an epitope.

    Returns
    -------
    df : pd.DataFrame
        Table of peptides that were selected with useful diagnostic stats about each selection."""
    
    """Using the same seed makes it reproducible"""
    np.random.seed(20110820)

    """Initialize monitoring object"""
    colOrder = ['peptide', 'pos', 'seqi', 'pos_count', 'delta_cov', 'count', 'freq', 'binder_count', 'binder_freq']
    outD = {k:[] for k in colOrder}
    
    """Reset the selction indices for all positions"""
    for vs in pa.pos:
        vs.selInds = []
        vs.sel = []
        vs.K = 0

    curCombCov = np.zeros(pa.Nkmers)
    for i in range(k):
        optimalVS = None
        optimalInd = None
        maxDeltaCov = 0
        """Go through the positions in a different random order each time to prevent bias"""
        randPosInd = np.random.permutation(len(pa.pos))
        for vsi in randPosInd:
            vs = pa.pos[vsi]
            if len(vs.sel) > 0:
                oldCov, _ = vs.computeCoverage(vs.selInds, mmTolerance=mmTolerance, epitopesOnly=epitopesOnly, bindingThreshold=bindingThreshold)
            else:
                oldCov = 0.
            
            needCovInd = find(~(vs.pwdist[:, vs.selInds] <= mmTolerance).any(axis=1))
            if epitopesOnly:
                binders = (vs.hlaGrid < bindingThreshold)
                """expressed_binder is shape (seqs,1) with values [0 or 1]"""
                expressed_binder = (binders * (vs.hlaMask > 0)).any(axis=1, keepdims=True)

                if len(vs.selInds) > 0:
                    """Think of this as just the numerator because you don't want to upweight selections that lead to high % coverage,
                    but low number of sequences/epitopes covered"""
                    oldCov = ((vs.pwdist[:, vs.selInds].min(axis=1) <= mmTolerance) * np.squeeze(expressed_binder)).sum()
                else:
                    oldCov = 0

                """How does each ind cover the peptides that need covering?"""
                pepCov = ((vs.pwdist[needCovInd,:] <= mmTolerance) * expressed_binder[needCovInd,:]).sum(axis=0)/expressed_binder[needCovInd,:].sum(axis=0, keepdims=True)
                seqi = argmax(pepCov)

                newCov = ((vs.pwdist[:, vs.selInds + [seqi]].min(axis=1) <= mmTolerance) * np.squeeze(expressed_binder)).sum()
            else:
                if len(vs.selInds) > 0:
                    oldCov = (vs.pwdist[:, vs.selInds].min(axis=1) <= mmTolerance).sum()
                else:
                    oldCov = 0
                pepCov = (vs.pwdist[needCovInd,:] <= mmTolerance).sum(axis=0)/len(needCovInd)
                """Argmax returns an ind across all seqs (columns, whereas rows were needed seqs with different indices)"""
                seqi = argmax(pepCov)
                newCov = (vs.pwdist[:, vs.selInds + [seqi]].min(axis=1) <= mmTolerance).sum()

            if (newCov - oldCov) > maxDeltaCov:
                optimalVS = vs
                optimalInd = seqi
                maxDeltaCov = newCov - oldCov
        """Select the peptide that was maximal"""
        optimalVS.setSelection(inds = optimalVS.selInds + [optimalInd])

        """Update the out dict"""
        vs = optimalVS
        outD['peptide'].append(vs.seqs[optimalInd])
        outD['pos'].append(pa.pos.index(vs))
        outD['pos_count'].append(len(vs.selInds))
        outD['seqi'].append(optimalInd)
        outD['delta_cov'].append(maxDeltaCov)
        identicalPepInd = vs.pwdist[:, optimalInd] == 0
        outD['count'].append(identicalPepInd.sum())
        outD['freq'].append(identicalPepInd.sum()/vs.pwdist.shape[0])
        outD['binder_count'].append((((vs.hlaGrid[identicalPepInd,:] < bindingThreshold) * vs.hlaMask[identicalPepInd,:]) > 0).any(axis=1).sum())
        outD['binder_freq'].append((vs.hlaMask[identicalPepInd,:] > 0).any(axis=1).sum()/vs.hlaMask.shape[0])

    pa.allocation = array([len(vs.selInds) for vs in pa.pos])
    outDf = pd.DataFrame(outD)[colOrder] 
    return outDf

def allMerBinders(align, hlaDf, ba, nmers=[9], bindingThreshold=np.log(500)):
    """Return a df of all nmers in the alignment along with start position and seq index"""
    align = padAlignment(align)
    cols = ['peptide', 'starti', 'seqi', 'L', 'count', 'binder_count', 'binder_seqs']
    outD = {k:[] for k in cols}
    for k in nmers:
        for seqi, (seq, hlas) in enumerate(zip(align, hlaDf.hlas)):
            for starti in range(len(seq)-k+1):
                gapped, mer = grabKmer(seq, starti, k)
                if not mer is None:
                    if not mer in outD['peptide']:
                        outD['peptide'].append(mer)
                        outD['starti'].append(starti)
                        outD['seqi'].append(align.index[seqi])
                        outD['L'].append(k)
                        outD['count'].append(0)
                        outD['binder_count'].append(0)
                        outD['binder_seqs'].append([])
                    
                    ind = outD['peptide'].index(mer)
                    outD['count'][ind] = outD['count'][ind] + 1
                    if not bindingThreshold is None and np.any(np.array([ba[(h, mer)] for h in hlas]) < bindingThreshold):
                        outD['binder_count'][ind] = outD['binder_count'][ind] + 1
                        outD['binder_seqs'][ind].append(align.index[seqi])
                else:
                    outD['peptide'].append(gapped)
                    outD['starti'].append(starti)
                    outD['seqi'].append(align.index[seqi])
                    outD['L'].append(0)
                    outD['count'].append(1)
                    outD['binder_count'].append(0)
                    outD['binder_seqs'].append([])
    return pd.DataFrame(outD)[cols]

def add10merCounts(hmers):
    """Modify binder status of 9mers in 'hmers' that are child peptides of 10mer binders"""
    out = hmers.copy()
    for i, row in out.loc[out.L==10].iterrows():
        if row['binder_count']>0:
            for seqi in row['binder_seqs']:
                for pep in [row['peptide'][1:], row['peptide'][:-1]]:
                    childInd = out.index[out.peptide == pep][0]
                    if not seqi in out.loc[childInd, 'binder_seqs']:
                        #print 'Added binder count to %s, child of %s' % (pep, row['peptide'])
                        out.loc[childInd, 'binder_seqs'].append(seqi)
                        out.loc[childInd, 'binder_count'] += 1
    return out

def coverageByStartPos(hmers, selectedPeptides, epitopeL, pwcov):
    """Tallies coverage of epitopes of length epitopeL in the
    peptide df hmers by peptides of arbitrary length in selectedPeptides.
    Depends on pre-computed coverage in df pwcov.

    Parameters
    ----------
    hmers : df
        A df of all peptides created by allMerBinders() from the alignment.
    selectedPeptides : list or pd.Series
    epitopeL : int
        Length of epitopes to be covered.
    pwcov : df
        Pre-computed coverage df computed by distance_df()
    """

    try:
        peptideL = len(selectedPeptides.iloc[0])
    except:
        peptideL = len(selectedPeptides[0])

    startPositions = list(range(hmers.starti.loc[hmers.L == epitopeL].max() + 1))
    cvec = np.nan * np.zeros(len(startPositions)) #fraction covered of all valid and invalid peptides
    epcvec = np.nan * np.zeros(len(startPositions)) #fraction of epitopes covered
    epdens = np.nan * np.zeros(len(startPositions)) #fraction of valid peptides that are epitopes
    validvec = np.nan * np.zeros(len(startPositions)) #number of peptides that are valid (do not start with gap)
    invalidvec = np.nan * np.zeros(len(startPositions)) #number of peptides that are invalid
    for i, starti in enumerate(startPositions):
        ind = (hmers.starti == starti) & (hmers.L == epitopeL)

        validvec[i] = hmers['count'].loc[ind].sum()
        invalidvec[i] = hmers['count'].loc[(hmers.starti == starti) & (hmers.L == 0)].sum()
        
        tot = validvec[i] + invalidvec[i]
        tmpCov = pwcov[selectedPeptides].loc[hmers.peptide.loc[ind].values].min(axis = 1).values

        eptot = hmers['binder_count'].loc[ind].sum()
        if eptot > 0:
            epcvec[i] = 1 - (tmpCov * hmers['binder_count'].loc[ind].values).sum() / eptot
        
        cvec[i] = 1 - (tmpCov * hmers['count'].loc[ind].values).sum() / tot
        if validvec[i] > 0:
            epdens[i] = eptot / validvec[i]
        
    return cvec, epcvec, epdens, validvec, invalidvec

def plotPTECoverage(coverageVars, sortBy='pos', titleStr=None):
    """Plot of coverage and epitope coverage.

    Parameters
    ----------
    coverageVars : tuple
        Output from coverageByStartPos.
    sortBy : string
        Use 'cov', 'epitope_cov', 'epitope_density' or None for start position."""

    def removeAxes(axh,remove=['right', 'top']):
        if 'right' in remove:
            axh.spines['right'].set_visible(False)
            axh.yaxis.set_ticks_position('left')
        if 'top' in remove:
            axh.spines['top'].set_visible(False)
            axh.xaxis.set_ticks_position('bottom')

    cvec, epcvec, epdens, validvec, invalidvec = coverageVars

    """Site vectors (sitevec), xticks (t) and xtick labels (tl)"""
    pctt = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    pcttl = (pctt * 100).astype(int)

    L = len(seqDf.seq.iloc[0])
    sitetl = np.concatenate(([1], np.arange(50, 550, 50), [L]))
    sitet = sitetl - 1

    """Make copies because some values may be changed for plotting"""
    cvec = cvec.copy()
    epcvec = epcvec.copy()
    epdens = epdens.copy()

    """Invalid peptides are counted as "covered" by default"""
    invalidFrac = invalidvec / (invalidvec + validvec)

    cvecNan = np.isnan(cvec)
    epcvecNan = np.isnan(epcvec)
    cvec[cvecNan] = 2
    epcvec[epcvecNan] = 2
    epdens[cvecNan] = 2

    L = cvec.shape[0]
    
    if sortBy is None or sortBy == 'pos':
        order = None
        ind = np.arange(len(cvec))
        pctAx = False
    elif sortBy[:3] == 'cov':
        order = ('cov', 'epitope_cov', 'epitope_density')
        pctAx = True
    elif sortBy == 'epitope_cov':
        order = ('epitope_cov', 'epitope_density', 'cov')
        pctAx = True
    elif sortBy[:12] == 'epitope_dens':
        order = ('epitope_density', 'epitope_cov', 'cov')
        pctAx = True
    else:
        print("Sort by what?")
        return
    
    if not order is None:
        sortArr = np.array([(x, y, z) for x, y, z in zip(cvec + invalidFrac, epcvec, epdens)], dtype=[('cov', 'float'), ('epitope_cov', 'float'), ('epitope_density', 'float')])
        ind = np.argsort(sortArr, order=order)[::-1]

    xvec = np.arange(L)
    
    plt.clf()
    axh1 = plt.subplot(3, 1, 1)
    cvec[cvecNan] = 1
    avg = np.mean(100 * (cvec[ind] + invalidFrac[ind]))
    if sortBy[:3] == 'cov':
        plt.annotate('%1.0f%%' % avg, xy=(L-1, avg), ha='right', va='bottom', size='x-large')
        plt.plot([0, L-1], [avg, avg], '--', color='black')
    cvec[cvecNan] = -5
    plt.fill_between(xvec, 100 * (cvec[ind] + invalidFrac[ind]), color='gray')
    plt.fill_between(xvec, 100 * cvec[ind], color='black', label='Coverage')
    
    plt.ylim((-5, 100))
    plt.xlim((0, L-1))
    plt.ylabel('Overall coverage (%)')
    removeAxes(axh1)
    if pctAx:
        plt.xticks(pctt*xvec[-1], pcttl)
    else:
        plt.xticks(sitet, sitetl)
    if not titleStr is None:
        plt.title(titleStr)
    
    axh2 = plt.subplot(3, 1, 2, sharex=axh1)
    epdens[cvecNan] = -5
    plt.fill_between(xvec, 100*epdens[ind], color='darkslateblue', label='Epitope density')
    plt.ylim((0, 100))
    plt.xlim((0, L-1))
    plt.ylabel('Sequences binding\nat least one HLA (%)')
    removeAxes(axh2)
    if pctAx:
        plt.xticks(pctt*xvec[-1], pcttl)
    else:
        plt.xticks(sitet, sitetl)

    axh3 = plt.subplot(3, 1, 3, sharex=axh1)
    epcvec[epcvecNan] = 1
    avg = np.mean(100*epcvec[ind])
    if sortBy == 'epitope_cov':
        plt.annotate('%1.0f%%' % avg, xy=(L-1, avg), ha='right', va='bottom', size='x-large')
        plt.plot([0, L-1], [avg, avg], '--', color='black')
    epcvec[epcvecNan] = -5
    plt.fill_between(xvec, 100*epcvec[ind], color='slateblue', label='Epitope coverage')
    plt.ylim((-5, 100))
    plt.xlim((0, L-1))
    plt.ylabel('Epitope coverage (%)')
    removeAxes(axh3)

    if pctAx:
        plt.xticks(pctt*xvec[-1], pcttl)
        plt.xlabel('Percentile of 9mers')
    else:
        plt.xticks(sitet, sitetl)
        plt.xlabel('Start position')

def brummeCoverageByStartPos(hmers, selectedPeptides, pwcov, seqDf, hlaDf):
    try:
        peptideL = len(selectedPeptides.iloc[0])
    except:
        peptideL = len(selectedPeptides[0])
    epitopeL = 9
    startPositions = list(range(hmers.starti.max() + 1))
    cvec = np.nan * np.zeros(len(startPositions))
    totvec = np.nan * np.zeros(len(startPositions))
    for i, starti in enumerate(startPositions):
        hmerSubIndex = hmers.index[hmers.starti == starti]
        tot = 0
        c = 0
        for hind in hmerSubIndex:
            pep = hmers.peptide.loc[hind]
            h = hmers.allele.loc[hind]
            #print hind, pep, h,
            #atLeast1 = False
            for seq, hlas in zip(seqDf.seq, hlaDf.hlas):
                twodigit = [tmp[:4].replace('*', '_') for tmp in hlas]
                if seq.find(pep) >= 0 and h[:4] in twodigit:
                    tot += 1
                    if pwcov[selectedPeptides].loc[pep].min() == 0:
                        #atLeast1 = True
                        c += 1
            #print atLeast1
        #tot = hmers['count'].loc[ind].sum()
        #tmpCov = pwcov[selectedPeptides].loc[hmers.peptide.loc[ind].values].min(axis = 1).values

        if tot > 0:
            cvec[i] = c / tot
            totvec[i] = tot
    
    return cvec, totvec

def plotPeptideCDF(align, hlaDf, ba, epitopeL=9, selected=[]):
    align = padAlignment(align)
    cols = ['peptide', 'starti', 'seqi', 'log-IC50', 'HLA']
    outD = {k:[] for k in cols}
    for seqi, (seq, hlas) in enumerate(zip(align, hlaDf.hlas)):
        for starti in range(len(seq)-epitopeL+1):
            gapped, mer = grabKmer(seq, starti, epitopeL)
            if not mer is None:
                outD['peptide'].append(mer)
                outD['starti'].append(starti)
                outD['seqi'].append(align.index[seqi])
                minIC50, minHLA = getIC50(ba, hlas, mer, nmer=[epitopeL], returnHLA=True)
                outD['log-IC50'].append(minIC50)
                outD['HLA'].append(minHLA)
    mersDf = pd.DataFrame(outD)[cols]
    mersDf['locus'] = mersDf.HLA.str.slice(0, 1)

    loci = mersDf.locus.unique()

    plt.clf()
    ic50Ticks = [20, 50, 200, 500, 1000, 2000, 5000, 10000, 20000]
    ECDF = sm.distributions.empirical_distribution.ECDF
    colors = palettable.colorbrewer.qualitative.Set1_3.mpl_colors
    for x, locus in enumerate(loci):
        ecdf = ECDF(mersDf['log-IC50'].loc[mersDf.locus == locus])
        plt.step(ecdf.x, ecdf.y, '-', color=colors[x], lw=3)

    for mer in selected:
        tmp = mersDf['log-IC50'].loc[mersDf.peptide == mer].values
        for i in range(tmp.shape[0]):
            plt.plot([tmp[i], tmp[i]], [0, 1], 'k-')

    plt.ylim((0, 1))
    colorLegend(colors=[colors[i] for i in range(len(loci))], labels=['HLA-%s' % h for h in loci])
    plt.xlim((0, 15))
    plt.xticks(np.log(ic50Ticks), ic50Ticks)
    plt.xlabel('Predicted HLA binding $IC_{50}$ (nM)')
    plt.ylabel('Fraction of peptides')

def mostSimilarComparator(row, comparator):
    pep = row['peptide']
    ssw = StripedSmithWaterman(query_sequence=pep,
                               protein=True,
                               substitution_matrix=_substMat)
    mx = None
    for pte in comparator:
        try:
            score = ssw(pte)['optimal_alignment_score']
        except ValueError:
            print('No alignment of %s and %s' % (pep, pte))
            raise
        if mx is None or score > mx['algn_score']:
            mx = {'algn_score':score, 'peptide':pte}
    return pd.Series(mx)
