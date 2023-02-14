import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import permutation, randint
import operator
import time
import sys

from pwseqdist import apply_pairwise_rect

"""TODO:
Write test of running select_ptes
Add code for computing coverage for one set of peptides for another. (Could be based on pwseqdist and coverage_distance I think)
Add code for plotting"""

def coverage_distance(epitope, peptide, mm_tolerance=1):
    """Determines whether pepitde covers epitope
    and can handle epitopes and peptides of different lengths.

    To be a consistent distance matrix:
        covered = 0
        not-covered = 1

    If epitope is longer than peptide it is not covered.
    Otherwise coverage is determined based on a mmTolerance

    Can accomodate strings or np.arrays (but not a mix).

    Parameters
    ----------
    epitope : str or np.array
    peptide : str or np.array
    mmTolerance : int
        Number of mismatches tolerated
        If dist <= mmTolerance then it is covered

    Returns
    -------
    covered : int
        Covered (0) or not-covered (1)"""

    tEpitope, tPeptide = type(epitope), type(peptide)
    assert tEpitope == tPeptide

    LEpitope, LPeptide = len(epitope), len(peptide)
    if LEpitope > LPeptide:
        return 1

    if isinstance(epitope, str):
        min_dist = np.array([np.sum([i for i in map(operator.__ne__, epitope, peptide[starti:starti+LEpitope])]) for starti in range(LPeptide-LEpitope+1)]).min()
    else:
        min_dist = np.array([(epitope != peptide[starti:starti+LEpitope]).sum() for starti in range(LPeptide-LEpitope+1)]).min()
    
    return 0 if min_dist <= mm_tolerance else 1


def select_ptes(epmers, pepmers, K=None, epitope_len=9, peptide_len=15, mm_tolerance=0):
    """Select peptides from pepmers that optimally cover the potential epitopes in epmers.

    Returns all peptides in order by the additional coverage they provide (descending)
    Also returns a vector of counts for the number of additional epitopes covered by each.

    This is essentially the PTE algorithm as developed by Fusheng Li, however
    there is no stopping threshold.

    Parameters
    ----------
    epmers_df : list
        Genetic sequences of epitopes to be covered and their count in the sequence set.
        Should include columns: sequence and count.
    pepmers_df : list
        Genetic sequences of peptides to select from that will cover epitopes.
        Should include columns: sequences, seqi, starti (for peptide index and/or peptide start position)
    K : int (default: None)
        Number of peptides to select. If None then select all peptides (in optimal order).
    mm_tolerance : int
        Coverage definition specifying the number of
        mismatches tolerated for coverage.

    Returns
    -------
    peptides : pd.DataFrame
        Peptides in pepmers_df in the order they were selected and the (marginal) coverage provided."""
    
    np.random.seed(110820)

    """Get the unique set of epitopes and peptides for selection"""
    epmers = np.unique(epmers_df['sequence'].values)

    pepmers = np.unique(pepmers_df['sequence'].values)

    if K is None:
        K = len(pepmers)

    """distance: covered = 0, not-covered = 1"""
    dmat = apply_pairwise_rect(metric=coverage_distance, seqs1=epmers, seqs2=pepmers, ncpus=1, args=(mm_tolerance,))
    
    """Weight the coverage matrix by it's frequency in the cohort"""
    cov_mat = (1 - dmat) * epmers_df['count'].values[:, None]

    """Use these "orig" copies to keep track of total coverage for each peptide.
    Non-orig matrices will be iteratively updated during the optimization to reflect delta coverage (ie the objective function)"""
    orig_cov_mat = cov_mat.copy()

    start_t = time.time()
    print(f'Selecting {K} peptides', end=' ')
    cols = ['peptide', 'upepi', 'seqi', 'starti', 'delta_count', 'total_count', 'delta_epcount', 'total_epcount']
    outD = {k:[] for k in cols}
    for i in range(K):
        if i % 50 == 0:
            print('.', end=' ')

        """NOTE: there are lots of ties here for argmax, so pick one at random."""
        tmp = cov_mat.sum(axis=0)
        mx_ind = np.where(tmp == tmp.max())[0]
        best_peptide = mx_ind[np.random.randint(len(mx_ind))]

        delta_coverage = cov_mat[:, best_peptide].sum()
        total_coverage = orig_cov_mat[:, best_peptide].sum()

        outD['peptide'].append(pepmers[best_peptide])
        outD['upepi'].append(best_peptide)
        outD['seqi'].append(pepmers_df['seqi'].iloc[best_peptide])
        outD['starti'].append(pepmers_df['starti'].iloc[best_peptide])
        outD['delta_count'].append(delta_coverage)
        outD['total_count'].append(total_coverage)
        
        """So that no future peptides will get credit for covering these epitopes"""
        covered_epitopes = cov_mat[:, best_peptide] > 0
        cov_mat[covered_epitopes,:] = 0
        cov_mat[:, best_peptide] = 0

    print(f'{(time.time() - startT)/60:1.1f} min')
    sys.stdout.flush()

    out = pd.DataFrame(outD)[cols]
    return out