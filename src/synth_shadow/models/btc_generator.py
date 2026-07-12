"""BTC 24h generator.

Generation plan:

1. Read the latest canonical Polygon BTC bars.
2. Compute 1h and 4h regime features.
3. Build future 5-minute timestamps for the next 24h.
4. Assign each future timestamp to a liquidity session.
5. Sample matching normalized historical session fragments.
6. Rescale with current volatility, vol-of-vol, volatility slope, momentum,
   and kurtosis.
7. Emit 1,000 paths with 289 prices anchored at the current BTC price.
"""
