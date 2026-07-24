#abm

Simple toy model representing agents who can invest in local businesses.

Entities: Investors, with location in space and wealth; and businesses that vary in profit and risk.
19x19 grid of businesses,  25 time steps

Processes: 1. Investor repositioning: Investors reposition to an adjacent path if they offer better trade-off of profit and risk. Only one investor can occupy a patch at a time. Executed at random order. 2. Accounting: Investors update their wealth, W=W+profit. With probability F, the wealth is set w=0.

Outputs: Mean wealth over time. Mean profit and risk chosen over time. 

Adaptive behavior: Agents decide to moth to the patch providing the highest value of an utility objective function $U = (W+TP)(1-F)^T$ , with T=5. 

Initial conditions: P and F are set randomly on (1, 10) and (0.01, 0.1) uniformly. 100 investors at random patches with W=0.  