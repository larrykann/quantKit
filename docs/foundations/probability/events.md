# Events

## Overview

The term "event" appears throughout probability theory but can mean different things depending on context. Understanding both the informal and formal definitions is essential for working with probabilistic models in finance.

## Informal Definition

In introductory probability discussions, an **event** is often described as an individual outcome or result of a random experiment. This colloquial usage treats events and outcomes as synonymous:

**Examples:**

- For coin toss: "Head" and "Tail" are events
- For stock prices: Each specific price ($150.25, $149.80, etc.) is an event  
- For dice roll: Each face (1, 2, 3, 4, 5, 6) is an event
- For portfolio returns: Each return value (+2.3%, -1.1%, etc.) is an event

This informal usage is intuitive and appears frequently in casual discussion, but it lacks the precision needed for rigorous mathematical analysis.

## Formal Mathematical Definition

In rigorous probability theory, an **event** is defined as any subset of the sample space:

$$E \subseteq \Omega$$

where $\Omega$ is the sample space containing all possible outcomes.

**Mathematical Notation:**

$$E \in \mathcal{F}$$

where $\mathcal{F}$ is the event space (sigma-algebra) containing all valid events.

### Examples of Formal Events

**Single coin toss** with $\Omega = \{\text{H}, \text{T}\}$:

- $\{\text{H}\}$ is an event (getting heads)
- $\{\text{T}\}$ is an event (getting tails)
- $\{\text{H}, \text{T}\}$ is an event (getting heads OR tails - certain event)
- $\emptyset$ is an event (impossible event)

**Two-coin toss** with $\Omega = \{(\text{H},\text{H}), (\text{H},\text{T}), (\text{T},\text{H}), (\text{T},\text{T})\}$:

- Event "at least one head": $E = \{(\text{H},\text{H}), (\text{H},\text{T}), (\text{T},\text{H})\}$
- Event "exactly one tail": $E = \{(\text{H},\text{T}), (\text{T},\text{H})\}$
- Event "both same": $E = \{(\text{H},\text{H}), (\text{T},\text{T})\}$

**Three-coin toss**:

- Event "having two tails": $E = \{(\text{H},\text{T},\text{T}), (\text{T},\text{H},\text{T}), (\text{T},\text{T},\text{H})\}$

## Events as Criteria

Events are typically associated with specific criteria or conditions. Computing probabilities involves filtering the sample space to find which outcomes satisfy the event's criteria.

**Process:**

1. Define the criterion (e.g., "having two tails")
2. Iterate through the sample space
3. Select all outcomes satisfying the criterion
4. The selected outcomes form the event set $E$
5. Calculate probability: $P(E) = \frac{|E|}{|\Omega|}$

**Example:** For three-coin toss, finding events with two tails:

- Sample space has 8 outcomes: $\{(\text{H},\text{H},\text{H}), (\text{H},\text{H},\text{T}), \ldots, (\text{T},\text{T},\text{T})\}$
- Apply criterion: "exactly two tails"
- Result: $E = \{(\text{H},\text{T},\text{T}), (\text{T},\text{H},\text{T}), (\text{T},\text{T},\text{H})\}$
- Probability: $P(E) = \frac{3}{8}$

## Why the Formal Definition Matters

The formal subset definition enables:

1. **Set operations on events**: Combine events using union ($\cup$), intersection ($\cap$), and complement operations
2. **Rigorous probability measures**: Define probability as a function $P: \mathcal{F} \to [0,1]$
3. **Complex criteria**: Work with events like "stock price between $100 and $150" representing infinitely many outcomes
4. **Event algebra**: Apply mathematical rules to compute probabilities of compound events

## Financial Applications

In quantitative finance, events typically represent complex conditions rather than single outcomes:

**Stock Price Events:**

- Event: "Stock price above $100" - represents infinitely many possible prices
- Event: "Price between $95 and $105" - interval of continuous values
- Event: "Daily return exceeds 5%" - derived criterion from price changes

**Portfolio Events:**

- Event: "Portfolio return between 5% and 10%" - range of outcomes
- Event: "At least 3 out of 5 stocks show positive returns" - compound condition
- Event: "Portfolio value exceeds target" - threshold-based

**Risk Management Events:**

- Event: "Daily loss exceeds VaR threshold" - tail risk
- Event: "Volatility spikes above historical average" - statistical criterion
- Event: "Correlation between assets exceeds 0.8" - relationship-based

## Reconciling Informal and Formal Usage

The informal usage (event = outcome) can be viewed as a special case of the formal definition:

- An **elementary event** or **singleton event** is a subset containing exactly one outcome
- When someone says "the event 'getting heads'", they formally mean the singleton set $\{\text{H}\}$
- For discrete sample spaces, every outcome corresponds to a singleton event

**Key distinction:**

- **Outcome**: Individual element $\omega \in \Omega$
- **Elementary event**: Singleton subset $\{\omega\} \subseteq \Omega$
- **Compound event**: Subset containing multiple outcomes $E \subseteq \Omega$ where $|E| > 1$

## Note on Terminology

Different texts and practitioners use "event" with varying levels of formality:

- **Informal contexts**: "Event" often means "outcome" (e.g., "the event of getting heads")
- **Formal mathematics**: "Event" always means "subset of sample space"
- **Practical finance**: Context determines usage; compound events are more common than elementary ones

In quantKit documentation, we use the **formal mathematical definition** (event as subset) unless explicitly stated otherwise. This maintains consistency with rigorous probability theory and statistical modeling.

## Computing Event Probabilities

For discrete finite sample spaces, probability of event $E$ is:

$$P(E) = \frac{|E|}{|\Omega|} = \frac{\text{number of outcomes in } E}{\text{total number of outcomes}}$$

For continuous sample spaces (like stock prices), events are typically intervals or regions, and probabilities are computed using probability density functions rather than counting.

## References

Nag, A. (2024). *Stochastic Finance with Python* (Chapter 3). Apress.

Papoulis, A., & Pillai, S. U. (2002). *Probability, Random Variables and Stochastic Processes* (4th ed.). McGraw-Hill.
