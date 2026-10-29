# Discrete Compound Interest

**Discrete compound interest** occurs when interest is calculated and added to the principal at specific intervals (daily, monthly, quarterly, annually). Unlike simple interest, each compounding period includes previously earned interest in the calculation base, creating exponential rather than linear growth.

## Mathematical Formula

$$A = P\left(1 + \frac{r}{m}\right)^{mt}$$

where:

- $A$ = Final amount after compound interest
- $P$ = Principal amount
- $r$ = Annual interest rate (as decimal)
- $m$ = Number of compounding periods per year
- $t$ = Time in years

## Key Characteristics

- Interest earns interest (compounding effect)
- Exponential growth pattern
- More frequent compounding yields higher returns
- Standard practice for most modern financial instruments

## Compounding Frequency Examples

- Annual: $m = 1$
- Semi-annual: $m = 2$ 
- Quarterly: $m = 4$
- Monthly: $m = 12$
- Daily: $m = 365$

## Financial Example

$8,000 invested at 6% annual rate compounded quarterly for 3 years:

$$A = 8,000\left(1 + \frac{0.06}{4}\right)^{4 \times 3} = 8,000(1.015)^{12} = \$9,564.92$$

Compare to simple interest: $8,000(1 + 0.06 \times 3) = \$9,440$

Compounding advantage: $124.92

## Modern Applications

- Savings accounts
- Certificates of deposit
- Corporate bonds
- Mortgage calculations (reverse application)
- Investment growth projections

The frequency of compounding significantly impacts returns. As compounding frequency approaches infinity, discrete compound interest converges to continuous compound interest.

## References

Fisher, I. (1930). *The Theory of Interest*. Macmillan.

Nag, A. (2024). *Stochastic Finance with Python* (Chapter 2). Apress.
