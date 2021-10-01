import json
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import softmax


np.set_printoptions(precision=2)
def complete_cdf(mean, stdev):
    cdf = np.array([])
    for i in range(0, 11):
        if i == 0:
            cdf = np.append(cdf, norm.cdf(i/10, loc=mean, scale=stdev))
        else:
            lower = norm.cdf(i/10-0.1, loc=mean, scale=stdev)
            upper = norm.cdf(i/10, loc=mean, scale=stdev)
            cdf = np.append(cdf, upper-lower)
    return cdf


def combine_distributions(*cdfs):
    cdf = cdfs[0]
    if len(cdfs) <= 1:
        return cdf
    else:
        for dist in cdfs[1:]:
            cdf += dist
        return cdf / len(cdfs)


if __name__ == '__main__':
    landmark_times = {}
    landmarks = json.load(open('../data/time_distributions.json'))
    for index, landmark in landmarks.items():
        distribution = landmarks[index]
        if distribution['dist1']['type'] == 'uniform':
            landmark_times[index] = np.ones(11)
            if index == 'library':
                landmark_times[index][8:11] = 0.
            elif index == 'printer':
                landmark_times[index][4] = 0.
                landmark_times[index][8:11] = 0.
        else:
            if len(distribution) == 1:
                distribution = distribution['dist1']
                dist = complete_cdf(distribution['mean'], distribution['stdev'])
                landmark_times[index] = dist
            else:
                distribution1 = distribution['dist1']
                dist1 = complete_cdf(distribution1['mean'], distribution1['stdev'])

                distribution2 = distribution['dist2']
                dist2 = complete_cdf(distribution2['mean'], distribution2['stdev'])

                landmark_times[index] = combine_distributions(dist1, dist2)

    trashcan_id = {}
    proximal_locations = json.load(open('../data/proximal_locations.json'))
    for k, v in proximal_locations.items():
        sums = np.sum([landmark_times[vi] for vi in v], axis=0)
        sums = np.exp(sums)
        sums = sums / np.linalg.norm(sums, ord=2)
        trashcan_id[k] = sums
        print(f'{k.ljust(40)}: {[round(v, 4) for v in trashcan_id[k]]}')
