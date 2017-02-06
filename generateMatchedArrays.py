import stellarTwins as st

thresholdSN = 0.001
filename = 'cutMatchedArrays.SN' + str(thresholdSN) +'.npz'
st.observationsCutMatched(SNthreshold=1., filename=filename)
