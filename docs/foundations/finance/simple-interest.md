# Simple Interest

**Simple interest** represents the most straightforward method of calculating interest earnings, where interest accrues only on the original principal amount. Unlike compound interest, previously earned interest does not generate additional interest, resulting in linear growth over time.

## Mathematical Formula

$$A = P(1 + rt)$$

where:
- $A$ = Final amount after interest
- $P$ = Principal (original amount invested)
- $r$ = Annual interest rate (expressed as decimal)
- $t$ = Time period in years

## Key Characteristics

- Interest calculation based solely on initial principal
- Linear growth pattern - interest amount remains constant each period
- No compounding effect
- Primarily used for short-term financial instruments

## Financial Example

Consider investing $5,000 in a simple interest certificate of deposit at 4% annual rate for 2 years:

$$A = 5,000(1 + 0.04 \times 2) = 5,000(1.08) = \$5,400$$

Total interest earned: $400 ($5,000 × 0.04 × 2)

## Implementation Examples

### Python

```python
def calculate_simple_interest(principal, rate, time):
   """Calculate final amount using simple interest formula"""
   return principal * (1 + rate * time)

def simple_interest_earned(principal, rate, time):
   """Calculate just the interest portion"""
   return principal * rate * time

# Example usage
principal = 5000.0
annual_rate = 0.04
years = 2.0

final_amount = calculate_simple_interest(principal, annual_rate, years)
interest_earned = simple_interest_earned(principal, annual_rate, years)

print(f"Principal: ${principal:,.2f}")
print(f"Interest earned: ${interest_earned:,.2f}")
print(f"Final amount: ${final_amount:,.2f}")
```

## Modern Applications

- Short-term promissory notes
- Some money market instruments
- Legal interest calculations
- Basic savings accounts (rare)

Most contemporary financial products use discrete compound interest or continuous compound interest due to their more favorable growth characteristics.

## References

Fisher, I. (1930). *The Theory of Interest*. Macmillan.

Nag, A. (2024). *Stochastic Finance with Python* (Chapter 2). Apress.
