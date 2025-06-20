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

## Implementation Examples

### Python

```python
def calculate_compound_interest(principal, annual_rate, compounding_frequency, years):
   """Calculate compound interest with discrete compounding periods"""
   periodic_rate = annual_rate / compounding_frequency
   total_periods = compounding_frequency * years
   return principal * ((1 + periodic_rate) ** total_periods)

def compound_interest_comparison(principal, annual_rate, years):
   """Compare different compounding frequencies"""
   frequencies = {
       'Annual': 1,
       'Semi-Annual': 2,
       'Quarterly': 4,
       'Monthly': 12,
       'Daily': 365
   }
   
   results = {}
   for name, freq in frequencies.items():
       final_amount = calculate_compound_interest(principal, annual_rate, freq, years)
       results[name] = final_amount
   
   return results

# Example usage
principal = 8000.0
annual_rate = 0.06
years = 3.0

results = compound_interest_comparison(principal, annual_rate, years)

print(f"Principal: ${principal:,.2f}")
print(f"Rate: {annual_rate:.1%} for {years} years\n")

for frequency, amount in results.items():
   interest_earned = amount - principal
   print(f"{frequency:12}: ${amount:8.2f} (Interest: ${interest_earned:6.2f})")
```

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
