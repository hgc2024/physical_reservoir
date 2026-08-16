# Mackey-Glass exploratory data analysis

## Reader's guide

Mackey-Glass data is a computer-generated sequence whose next value depends on both its current value and a delayed past value. It is useful here because it creates bounded but difficult-to-predict oscillations. The goal of this report is to check the generated data before it is used to train a forecasting model; no physical-computing background is required.

## What was generated

- **Samples analyzed:** 5,000
- **Integration interval:** 0.1 model time units per sample
- **Discarded startup period:** 1,000 samples (100 time units)
- **Delayed feedback:** 17 time units
- **Equation settings:** feedback `a=0.2`, decay `b=0.1`, nonlinear exponent `gamma=10`

The startup period is discarded because the sequence begins from an artificial constant history. Removing it gives the system time to settle into its characteristic dynamics.

## What the values look like

- **Observed range:** 0.4168 to 1.3181
- **Mean (arithmetic average):** 0.9304
- **Median (middle value):** 0.9721
- **Standard deviation (typical spread around the mean):** 0.2257
- **Middle 50% of observations:** 0.7476 to 1.1246

The values are finite, bounded, and visibly variable rather than constant. This is the basic behavior needed for the planned forecasting experiments.

## How values depend on earlier values

Autocorrelation measures how similar the sequence is to a time-shifted copy of itself. A value near `1` means strong similarity, `0` means little linear similarity, and a negative value means the shifted patterns tend to move in opposite directions.

- **One-step autocorrelation:** 1.000
- **First zero crossing:** 12.4 time units
- **Strongest negative relationship inspected:** correlation -0.730 at 24.1 time units
- **Later recurrence:** A later positive relationship appears around 49.2 time units (correlation 0.64).

This rise-and-fall pattern confirms strong temporal structure: nearby samples are not independent, and relationships persist over many steps. It supports chronological evaluation and makes random shuffling inappropriate. Autocorrelation alone does not prove chaos or establish the best forecasting horizon.

## How the data was prepared

The sequence was kept in its original order and divided into:

- **Training:** 3,500 samples — used to fit model parameters
- **Validation:** 750 samples — reserved for choosing model settings
- **Test:** 750 samples — reserved for one final, unbiased evaluation

Standardization used only the training segment, with training mean `0.9200` and scale `0.2291`. The same values were then applied to validation and test segments. This avoids using information from the future during training.

As a check, the standardized training data has mean `-1.74e-07` and standard deviation `1.000`. Predictions can later be converted back to the original scale using the saved mean and scale.

## Practical conclusion

The generated series passed the initial data-quality checks: it contains no missing or infinite values, has meaningful variation, and exhibits long temporal dependence. It is ready for the next methodology stage: defining forecasting inputs and targets before driving the reservoir model.
