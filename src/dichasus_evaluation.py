#!/usr/bin/env python3

import numpy as np

def compute_localization_metrics(positions_estimated, position_labels):
    errorvectors = position_labels - positions_estimated
    errors = np.sqrt(errorvectors[:,0]**2 + errorvectors[:,1]**2)
    mae = np.mean(errors)
    r90 = np.percentile(errors, 90)

    return mae, r90
