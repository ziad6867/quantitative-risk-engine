# Interactive Liability-Driven Investing (LDI) & Dynamic Risk Engine



### The Elevator Pitch
A full-stack, real-time quantitative finance application that bridges the gap between institutional risk management theory and applied computer engineering. It features dynamic portfolio optimization, tail risk diagnostics, and a trend-following Portfolio Insurance (CPPI) algorithm.

### Executive Summary
Most financial models exist as static, brittle scripts. This project translates rigorous academic quantitative finance into a production-grade, interactive web application. Built entirely in Python, the engine dynamically ingests over a decade of live market data to simulate complex portfolio behaviors under extreme market stress. By decoupling the mathematical backend (handling SciPy optimization and pandas vectorization) from a custom-styled Streamlit frontend, the architecture mirrors how modern fintech platforms build internal risk tools. 

### Technical Architecture & Stack
*   **Backend Math & Logic:** Custom `edhec_risk_kit` utilizing `SciPy` (SLSQP minimization for optimal weights) and `NumPy`/`Pandas` (vectorized matrix operations, covariance calculations, and rolling signal generation).
*   **Data Pipeline:** Live integration with the `yfinance` API, complete with automated structural alignment and NaN-handling to ensure millisecond-fast processing of mismatched time-series data.
*   **Frontend UI:** `Streamlit` with custom CSS injection, breaking away from standard corporate templates to deliver a high-contrast, gaming-inspired interface with vibrant Plotly charting. 

### Core Engine Capabilities

#### 1. Modern Portfolio Theory (MPT) & Optimization
The engine calculates the annualized returns and covariance matrix of any given asset basket to dynamically plot the Markowitz Efficient Frontier. Using SciPy's Sequential Least Squares Programming, the algorithm mathematically pinpoints and visualizes the exact asset allocations required to achieve the Maximum Sharpe Ratio (MSR) and the Global Minimum Variance (GMV).

#### 2. Liability-Driven Investing (LDI) Simulator
Moving beyond simple asset growth, this module simulates a $1,000,000 institutional portfolio matched against long-term fixed-income liabilities. It tracks cumulative returns and plots a live Funding Ratio timeline to visualize exactly how well the growth assets cover the liability floor over time. 

#### 3. Tail Risk & Non-Normal Distribution Diagnostics
Financial markets do not follow a perfect bell curve. This diagnostic suite visualizes the "fat tails" of market crashes. It calculates the historical drawdown series (rendered via an interactive underwater plot) alongside critical risk metrics including the 5% Historic VaR, Modified Cornish-Fisher VaR, and Conditional Value at Risk (Expected Shortfall).

#### 4. Trend-Following CPPI (Constant Proportion Portfolio Insurance)
An interactive algorithmic trading simulator that runs a daily loop balancing a risky asset against a safe asset to guarantee the portfolio never breaches a hard, user-defined wealth floor. To prevent the "whipsaw" effect during volatile markets, the mathematical floor is augmented with a dynamic financial signal processing filter—a 50/200-day Simple Moving Average (SMA) crossover system that automatically detects negative momentum and cuts risky exposure before a crash occurs.

---

### How to Run Locally

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/quantitative-risk-engine.git](https://github.com/YOUR_USERNAME/quantitative-risk-engine.git)
