# Ai-cup/E.ON day ahead smart meter forecasting

## General Setting

Electricity, till date, cannot be stored in large amounts, therefore supply and demand need to always be balanced by the energy providers. The accurate short term forecast of energy demand is critical for the operations and control of productive capacity, with significant consequences. However, as with any forecast, there is typically some uncertainty involved. This uncertainty is especially heightened in the case of energy forecasting today, where alternate sources of energy such as solar panels are ubiquitous. Also, with the transition to e-mobility additional non-traditional consumer patterns contribute to the forecasting uncertainty. Therefore, understanding electricty consumption behaviour either for individual households or for regional groups of households becomes key for the future electricity market.

## The Task

This data science challenge task entails estimating day-ahead-forecasts for upto a week, for 61 groups of dwellings in the UK energy market, based on geographical similarity. The challenge has two sub-tasks- the first where only one value for the single day ahead is required to be estimated, in other words-the aggregated day-ahead demand. In the second sub task, the demand for each hour in the day-ahead is to be estimated (24 per day).You are provided with historical half-hourly energy readings for the 61 anonymised groups between 1 January, 2017 and 04 September 2019. A week is sliced off from each 45 day window and reserved for testing purposes. You are required to estimate these missing periods in the two frequencies. Every group consists of a different number of dwellings, which energy consumption profile has been summed up for two reasons: data privacy and forecasting accuracy. All data is provided in csv format and described below. We also provide code snippets for loading the data and creating submission files.

## Project structure 

In this project, we were working in several directions to find the best way to solve this problem. Linear regression as a baseline, LSTM as a neural network, Merlion as an ensemble tool, and Transformer. Different approaches are placed in different folders, according to the name.

