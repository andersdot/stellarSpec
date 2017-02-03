import stellarTwins as st

thresholdSN = 0.001
filename = 'cutMatchedArrays.SN' + str(thresholdSN) +'.npz'
observationsCutMatched(SNthreshold=1., filename=filename)
