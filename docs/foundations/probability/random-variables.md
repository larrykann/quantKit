# Random Variables

Random variables provide a well-defined quantification scheme for mapping from [[Events (Probability Theory)|events]] to numbers. They transform qualitative outcomes into quantitative measures, making probability calculations manageable.

**Formal Definition:**
$$X(\omega): \Omega \to \mathbb{R}, \quad \omega \in \mathbb{F}$$

where $X$ is defined as a function over event $\omega$, mapping from the [[Sample Space (Ω)|sample space]] $\Omega$ to the real numbers $\mathbb{R}$.

**Purpose and Importance:**

- Convert descriptive events into numerical values
- Enable mathematical operations on probabilistic outcomes
- Provide a systematic way to work with uncertain quantities
- Form the foundation for statistical analysis and modeling

**Examples:**

- Coin toss: Map {Head, Tail} → {1, 0}
- Stock price: Map "Google stocks reaching $135" → $X = 138.49$
- Dice roll: Map face outcomes → {1, 2, 3, 4, 5, 6}

**Financial Applications:**
For financial modeling, security prices are treated as random variables with real, non-negative values. Numbers represented by random variables should be chosen judiciously to have meaningful representation for the specific financial context.

## References

Nag, A. (2024). *Stochastic Finance with Python* (Chapter 3). Apress.

Papoulis, A., & Pillai, S. U. (2002). *Probability, Random Variables and Stochastic Processes* (4th ed.). McGraw-Hill.
