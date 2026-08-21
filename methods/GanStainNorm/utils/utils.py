"""General utility funcitons"""

from typing import List, Sequence, Tuple

def get_paired_samples(
        unstained_list: Sequence,
        stained_list: Sequence
    ) -> List[Tuple[str, str]]:
    """Pair sample (WSI) names based on their prefix"""
    unstained_list.sort()
    stained_list.sort()
    paired_sample_list: List[Tuple[str, str]] = []
    for unstained, stained in zip(unstained_list, stained_list):
        index_unstained = unstained.split('_')[1]
        index_stained = stained.split('_')[1]
        if index_unstained != index_stained:
            raise RuntimeError('Config does not contain paired data')
        paired_sample_list.append((unstained, stained))
    return paired_sample_list
