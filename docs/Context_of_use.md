# Context of use

This document describes the context of use of BoneStrengthML, the case study to which the pipeline in this repository is applied.
It is the source of the thresholds in `config/BoneStrengthML.yml`.

## Application description

BoneStrengthML is intended as an ML-based surrogate of BoneStrength, a complex and resource-demanding biophysical model based on finite element (FE) method.
BoneStrengthML performs a regression task that predicts bone strain values – induced in the subject bone for simulated fall events – based on patient-specific femur characteristics.

The model takes inputs derived from CT scans, including:

- 93 principal components representing a morpho-densitometric femur statistical atlas (i.e., a summary of information concerning the geometry of the femur and its mineral density sampled over a cartesian lattice)
- PosAnt and MedLat angles describing impact force orientation during side-fall events

The model outputs two critical strain measurements:

- `maxStrain_11`: maximum first principal strain from finite-element simulation
- `maxStrain_33`: absolute value of minimum third principal strain from finite-element simulation


## Error threshold (validation)

The strain values which have been established as thresholds to estimate femur fracture are 10400 &mu;&epsilon; and 7300 &mu;&epsilon; in compression and tension respectively.
The accuracy of the FE model BoneStrengthML aims to surrogate was found to be 7% (RMSE% normalized to max value).
This means that the FE model has an error equal to 511 &mu;&epsilon; in predicting tensile strains (`maxStrain_11`) and equal to 729 &mu;&epsilon; in predicting compressive strain (`maxStrain_33`).
As BoneStrengthML should surrogate this FE model we expect the acceptability threshold to be one order of magnitude smaller: 51 &mu;&epsilon; and 73 &mu;&epsilon; for `maxStrain_11` and `maxStrain_33`, respectively.
Since in no case we will accept errors higher than the threshold, we will use infinity norm of the error.


## Verification methodology and thresholds

In order to have an acceptable coverage of the 95-dimensional space, we require at least 1 M samples in the verification process.

For convergence we accept a relative error in the output with respect to the full-trained model of 5%.

To limit error propagation, we require that the relative error of any input is not amplified by the ML model.
We require to test 40% of the admissible input space and to perturb each of the inputs by 1/1000 of its admissible range.

We expect the numerical error to be one order of magnitude below the ML model error with respect to the FE model (i.e., the validation error), so we require that the standard deviations of the outputs are 5.1 &mu;&epsilon; for `maxStrain_11` and 7.3 &mu;&epsilon; for `maxStrain_33`, respectively.
We require to use at least 10 repeated trainings for a reliable estimation of the numerical error.


## Uncertainty quantification

Since BoneStrengthML is a surrogate model of a simulation, there are no error associated to the input values, so uncertainty quantification was deemed not relevant.
