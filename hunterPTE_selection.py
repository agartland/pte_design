from hunterPTE_load import *

exec(mycompile('hunterPTE_funcs.py'))

"""python iedb_predict.py --method NetMHCpan --pep ~/gitrepo/peptidedesign/data/cPTE_design/gag_13Aug2018.9.mers --hla ~/data/HunterPTE/hunter_jan2015.hla --out ~/data/HunterPTE/gag_13Aug2018.9.out --cpus 4"""

def modSelectPTEs(epmersDf,
                  pepmersDf,
                  N,
                  K=None,
                  epitopeL=9,
                  peptideL=15,
                  mmTolerance=0,
                  epitopesOnly=False):
    sel = selectPTEs(epmersDf,
                     pepmersDf,
                     K=K,
                     epitopeL=epitopeL,
                     peptideL=peptideL,
                     mmTolerance=mmTolerance,
                     epitopesOnly=epitopesOnly)
    sel['delta_freq'] = sel.delta_count/N
    sel['total_freq'] = sel.total_count/N
    sel['delta_epfreq'] = sel.delta_epcount/N
    sel['total_epfreq'] = sel.total_epcount/N
    return sel

epitopeL = 9
mmTolerance = 0
K = 290
bindingThreshold = 10000
peptideL = 10

hmers = allMerBinders(seqDf.seq,
                      hlaDf,
                      ba,
                      nmers=[epitopeL, 10], 
                      bindingThreshold=np.log(bindingThreshold))
#hmers = add10merCounts(hmers)

epitope_seqs = seqs2mat(hmers.peptide.loc[hmers.L == epitopeL].values)

for epitopesOnly in [False, True]:
    for peptideL in [10, 15]:
        peps = align2mers_tracked(seqDf.seq, nmers=[peptideL])
        """Discard "forbidden" peptides from list of possible
        peptide selections before optimizing."""
        peps = peps.loc[~peps.peptide.map(isforbiddenPep)]

        peptide_seqs = seqs2mat(peps.peptide.loc[peps.L == peptideL].values)
        pwcov = distance_df(epitope_seqs,
                            peptide_seqs,
                            metric=nbmetrics.nb_coverage_distance,
                            args=(mmTolerance,),
                            symetric=False)
        selection = modSelectPTEs(hmers,
                                  peps,
                                  N=seqDf.shape[0],
                                  K=K,
                                  epitopeL=epitopeL,
                                  peptideL=peptideL,
                                  mmTolerance=mmTolerance,
                                  epitopesOnly=epitopesOnly)
        """Add extra columns for interpretation"""
        selection['HXB2Start'] = selection['peptide'].map(hxb2AlignFunc)
        selection['HXB2Peptide'] = selection.apply(getHXB2Peptide, axis=1)
        selection['ConsensusPeptide'] = selection.apply(getConsensusPeptide, axis=1)
        selection['ConsensusHamming'] = selection.apply(lambda r: hamming_distance(r['peptide'], r['ConsensusPeptide']), axis=1)
        selection['HLABinding500'] = selection.apply(partial(predictedHLAs, bindingThreshold=np.log(500)), axis=1)
        selection['HLABinding10000'] = selection.apply(partial(predictedHLAs, bindingThreshold=np.log(10000)), axis=1)
        selection['nonMatchedPTID-HLA'] = selection.apply(partial(predictedPTIDs, bindingThreshold=np.log(10000)), axis=1)
        selection['nonMatchedPTID'] = selection.apply(partial(matchedPTIDs), axis=1)
        selection['FractionConsensus'] = selection.apply(fractionConsensus, axis=1)

        tmp = selection.apply(partial(mostSimilarComparator, comparator=ptegDf.peptide), axis=1)
        tmp = tmp.rename(lambda col: 'PTEg_%s' % col, axis=1)
        selection = pd.concat((selection, tmp), axis=1)
        
        coverageVars = coverageByStartPos(hmers, selection.peptide, epitopeL, pwcov)

        fileArgs = (K, mmTolerance, bindingThreshold, peptideL, epitopesOnly)
        annParams = dict(s='K=%d, MM=%d, BT=%d, PEP=%d, EP=%d' % fileArgs,
                         xy=(0.02, 0.98),
                         xycoords='figure fraction',
                         ha='left',
                         va='top',
                         size='x-large')
        fileArgs = (K, mmTolerance, bindingThreshold, peptideL, epitopesOnly)
        filename = opj(GIT_PATH, 'peptidedesign', 'data', 'selection_K%d_MM%d_BT%d_PEP%d_EP%d.csv' % fileArgs)
        selection.sort_values(by='starti').to_csv(filename)
        for sb in ['cov', 'epitope_cov', 'pos']:
            plt.figure(1, figsize=(10, 11.8))
            plotPTECoverage(coverageVars, sortBy=sb)
            plt.annotate(**annParams)
            filename = opj(GIT_PATH, 'peptidedesign', 'figures', 'PTE_K%d_MM%d_BT%d_PEP%d_EP%d_%s.png' % (fileArgs + (sb,)))
            plt.figure(1).savefig(filename)

"""Make a comparison plot showing coverage of overlapping 10mers as PTE"""
for peptideL, overlap in [(15, 11), (10, 8)]:
    conSeq = consensus(seqDf.seq)
    """Removing gaps before designing consensus peptides. STARTI will no longer match the alignment"""
    conSeq = conSeq.replace('-', '')
    conMers, starti = overlappingKmers(conSeq, k=peptideL, overlap=overlap, returnStartInds=True)
    conDf = pd.DataFrame({'peptide':conMers,'starti':starti})
    if peptideL == 10:
        """Many (59 of 279) of the conMers don't appear in hmers due to gaps.
        IOW they don't appear in any sequence!"""
        conDf['notInCohort'] = ~conDf.peptide.isin(hmers.peptide.values)
        """For synthesizing don't discard, fix."""
        fixFunc = lambda starti: fixforbiddenPep(consensus(seqDf.seq), 10, starti)
        conDf['fixed'] = conDf.starti.map(fixFunc)
        """Of the remaining 220 conMers, only 60 appear in the epitope-based selection of K=300"""
        conDf['inCohortPTE'] = conDf.peptide.isin(selection.peptide.values)
        
        """Remove one peptide that could not be "fixed"""
        conDf = conDf.loc[conDf.fixed.str.len()<20]

        tmp = conDf.apply(partial(mostSimilarComparator, comparator=selection.peptide), axis=1)
        tmp = tmp.rename(lambda col: 'coh_pte_%s' % col, axis=1)
        conDf = pd.concat((conDf, tmp), axis=1)

        cols = ['peptide', 'fixed', 'starti', 'inCohortPTE', 'coh_pte_algn_score', 'coh_pte_peptide']
        conDf[cols].to_csv(opj(GIT_PATH, 'peptidedesign', 'data', 'cohort_consensus%dmers.csv' % peptideL))

    """Discard "forbidden" peptides for this comparison."""
    conMers = [p for p in conMers if not isforbiddenPep(p)]
    pwcov = distance_df(epitope_seqs,
                        seqs2mat(conMers),
                        metric=nbmetrics.nb_coverage_distance,
                        args=(mmTolerance,),
                        symetric=False)

    coverageVars = coverageByStartPos(hmers, conMers, epitopeL, pwcov)
    plt.figure(3, figsize = (10, 11.8))
    tit = 'Coverage by %d overlapping consensus %dmers (%d AA overlap)' % (len(conMers), peptideL, overlap)
    plotPTECoverage(coverageVars, sortBy='epitope_cov', titleStr=tit)
    plt.figure(3).savefig(opj(GIT_PATH, 'peptidedesign', 'figures', 'coverage_by_con%dmers_byEpitope.png' % (peptideL)))
    plotPTECoverage(coverageVars, sortBy='pos', titleStr=tit)
    plt.figure(3).savefig(opj(GIT_PATH, 'peptidedesign', 'figures', 'coverage_by_con%dmers_byStartPos.png' % (peptideL)))
    plotPTECoverage(coverageVars, sortBy='cov', titleStr=tit)
    plt.figure(3).savefig(opj(GIT_PATH, 'peptidedesign', 'figures', 'coverage_by_con%dmers_byCov.png' % (peptideL)))


"""Compare to actual consensus 15mers used for mapping"""
cliveAlignFn = opj(dataFolder, 'Peptide positons relative to 500aa HBX2_CM.xlsx')
cmDf = pd.read_excel(cliveAlignFn).iloc[:, :6]
uConsensusPeptides = cmDf.loc[cmDf.PepSet == 'Consensus', 'Sequence'].tolist()
uConsensusPeptides = [p for p in uConsensusPeptides if len(p) == 15]
pwcov = distance_df(epitope_seqs,
                    seqs2mat(uConsensusPeptides),
                    metric=nbmetrics.nb_coverage_distance,
                    args=(mmTolerance,),
                    symetric=False)

coverageVars = coverageByStartPos(hmers, uConsensusPeptides, epitopeL, pwcov)
plt.figure(3, figsize = (10, 11.8))
tit = 'Coverage by %d overlapping consensus 15mers (actual)' % (len(uConsensusPeptides))
plotPTECoverage(coverageVars, sortBy='epitope_cov', titleStr=tit)
plt.figure(3).savefig(opj(GIT_PATH, 'peptidedesign', 'figures', 'coverage_by_actcon15mers_byEpitope.png'))
plotPTECoverage(coverageVars, sortBy='pos', titleStr=tit)
plt.figure(3).savefig(opj(GIT_PATH, 'peptidedesign', 'figures', 'coverage_by_actcon15mers_byStartPos.png'))
plotPTECoverage(coverageVars, sortBy='cov', titleStr=tit)
plt.figure(3).savefig(opj(GIT_PATH, 'peptidedesign', 'figures', 'coverage_by_actcon15mers_byCov.png'))


"""Compare to coverage by PTE globals"""
dataFolder = opj(GIT_PATH, 'peptidedesign','data')
peptidesFn = opj(dataFolder, 'all_peptides_29Nov2017.csv')
pepDf = pd.read_csv(peptidesFn)
PTEg = pepDf.loc[pepDf.PeptideSet=='Global'].Sequence.dropna().unique()

#PTEg = [p for p in PTEg if not isforbiddenPep(p)]

pwcov = distance_df(epitope_seqs, seqs2mat(PTEg), metric=nbmetrics.nb_coverage_distance, args=(mmTolerance,), symetric=False)
coverageVars = coverageByStartPos(hmers, PTEg, epitopeL, pwcov)
plt.figure(3, figsize = (10, 11.8))
tit = 'Coverage by %d Global PTE 15mers' % (len(PTEg))
plotPTECoverage(coverageVars, sortBy='epitope_cov', titleStr=tit)
plt.figure(3).savefig(opj(GIT_PATH, 'peptidedesign', 'figures', 'coverage_by_PTEg_byEpitope.png'))
plotPTECoverage(coverageVars, sortBy='pos', titleStr=tit)
plt.figure(3).savefig(opj(GIT_PATH, 'peptidedesign', 'figures', 'coverage_by_PTEg_byStartPos.png'))
plotPTECoverage(coverageVars, sortBy='cov', titleStr=tit)
plt.figure(3).savefig(opj(GIT_PATH, 'peptidedesign', 'figures', 'coverage_by_PTEg_byCov.png'))

"""Evaluate lower K for 65% coverage"""
K = 100
peps = align2mers_tracked(seqDf.seq, nmers=[peptideL])
"""Discard "forbidden" peptides from list of possible
peptide selections before optimizing."""
peps = peps.loc[~peps.peptide.map(isforbiddenPep)]

peptide_seqs = seqs2mat(peps.peptide.loc[peps.L == peptideL].values)
pwcov = distance_df(epitope_seqs,
                    peptide_seqs,
                    metric=nbmetrics.nb_coverage_distance,
                    args=(mmTolerance,),
                    symetric=False)
selection = modSelectPTEs(hmers,
                          peps,
                          N=seqDf.shape[0],
                          K=K,
                          epitopeL=epitopeL,
                          peptideL=peptideL,
                          mmTolerance=mmTolerance,
                          epitopesOnly=epitopesOnly)
"""Add extra columns for interpretation"""
selection['HXB2Start'] = selection['peptide'].map(hxb2AlignFunc)
selection['HXB2Peptide'] = selection.apply(getHXB2Peptide, axis=1)
selection['ConsensusPeptide'] = selection.apply(getConsensusPeptide, axis=1)
selection['ConsensusHamming'] = selection.apply(lambda r: hamming_distance(r['peptide'], r['ConsensusPeptide']), axis=1)
selection['HLABinding500'] = selection.apply(partial(predictedHLAs, bindingThreshold=np.log(500)), axis=1)
selection['HLABinding10000'] = selection.apply(partial(predictedHLAs, bindingThreshold=np.log(10000)), axis=1)
selection['nonMatchedPTID-HLA'] = selection.apply(partial(predictedPTIDs, bindingThreshold=np.log(10000)), axis=1)
selection['nonMatchedPTID'] = selection.apply(partial(matchedPTIDs), axis=1)
selection['FractionConsensus'] = selection.apply(fractionConsensus, axis=1)

tmp = selection.apply(partial(mostSimilarComparator, comparator=ptegDf.peptide), axis=1)
tmp = tmp.rename(lambda col: 'PTEg_%s' % col, axis=1)
selection = pd.concat((selection, tmp), axis=1)

coverageVars = coverageByStartPos(hmers, selection.peptide, epitopeL, pwcov)

fileArgs = (K, mmTolerance, bindingThreshold, peptideL, epitopesOnly)
annParams = dict(s='K=%d, MM=%d, BT=%d, PEP=%d, EP=%d' % fileArgs,
                 xy=(0.02, 0.98),
                 xycoords='figure fraction',
                 ha='left',
                 va='top',
                 size='x-large')
fileArgs = (K, mmTolerance, bindingThreshold, peptideL, epitopesOnly)
filename = opj(GIT_PATH, 'peptidedesign', 'data', 'selection_K%d_MM%d_BT%d_PEP%d_EP%d.csv' % fileArgs)
selection.sort_values(by='starti').to_csv(filename)
for sb in ['cov', 'epitope_cov', 'pos']:
    plt.figure(1, figsize=(10, 11.8))
    plotPTECoverage(coverageVars, sortBy=sb)
    plt.annotate(**annParams)
    filename = opj(GIT_PATH, 'peptidedesign', 'figures', 'PTE_K%d_MM%d_BT%d_PEP%d_EP%d_%s.png' % (fileArgs + (sb,)))
    plt.figure(1).savefig(filename)

"""Compare against LANL A-list peptides"""
adf, bdf = HIVABlist.loadEpitopes(opj(GIT_PATH, 'HIVABlist', 'data'))
adf, bdf, abPred = HIVABlist.loadPredictions(adf, bdf, opj(GIT_PATH, 'HIVABlist', 'data', 'predictions.csv'))

adf = adf.loc[(adf.Protein=='Gag') & (adf.Epitope.map(len) <= 10) & (adf.Epitope.map(isvalidmer))]

plt.figure(12, figsize=(15, 6))
HIVABlist.plotEpitopeCDF(adf, abPred)
plt.title('HIV-1 epitopes: CTL A-list')
plt.figure(12).savefig(opj(GIT_PATH, 'peptidedesign', 'figures', 'LANL_Alist_IC50_CDF.png'))

adf = alignKnownEpitopes(adf, consensus(seqDf.seq))
adf = adf.drop_duplicates(subset='conEpitope')

selection_seqs = seqs2mat(selection.peptide.values)
def iscovered(pep):
    known_seq = seqs2mat([pep])
    pwcov = distance_df(known_seq,
                        selection_seqs,
                        metric=nbmetrics.nb_coverage_distance,
                        args=(mmTolerance,),
                        symetric=False)
    return np.any(pwcov.values == 0, axis=1)[0]

adf['coveredByCohortPTE'] = adf.conEpitope.map(iscovered)
adf['inCohortPTE'] = adf.conEpitope.isin(selection.peptide.values)
cols = ['conEpitope',
        'HXB2 start',
        'HXB2 end',
        'Subprotein',
        'HLA',
        'sitei',
        'hdist',
        'coveredByCohortPTE',
        'inCohortPTE']
adf[cols].sort_values(by='sitei').to_csv(opj(GIT_PATH, 'peptidedesign', 'data', 'LANL_Alist_peptides.csv'))