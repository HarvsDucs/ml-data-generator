
## 🎬 Demo

<video src="ml-data-generator-demo.mp4" width="800" controls></video>

### 1. Statistical Distributions
- **Normal (Gaussian) Distribution**: Generate features with specific means and standard deviations
- **Uniform Distribution**: Create random noise or non-predictive features
- **Power Law (Long-tail)**: Simulate real-world skewed data (e.g., wealth, city populations)

### 2. Linear vs Non-Linear Relationships
- **Linear Data**: Perfect for testing linear regression models
- **Polynomial Relationships**: Quadratic, cubic, and quartic functions
- **Interaction Effects**: Multi-variable dependencies for tree-based model practice

### 3. Class Imbalance
- Generate imbalanced binary classification datasets
- SMOTE-like oversampling for synthetic minority samples
- Perfect for fraud detection, rare disease diagnosis scenarios

### 4. Noise and Outliers
- **Gaussian Noise**: Add random variation to clean data
- **Label Noise**: Flip labels to test model robustness
- **Outliers**: Inject extreme values to test robust scalers

### 5. Time-Series Variations
- **Trend Analysis**: Linear and exponential trends
- **Seasonality**: Periodic patterns
- **Stationarity**: Compare stationary vs non-stationary data

### 6. Categorical Data
- Basic categorical features for encoding practice
- Faker-generated realistic data (names, emails, addresses)
- High-cardinality features for advanced encoding techniques

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ml-data-generator.git
cd ml-data-generator

# Install dependencies with uv (creates .venv automatically)
uv sync
```

### Running the App

```bash
uv run streamlit run data_generator.py
```

The app will open in your browser at `http://localhost:8501`

## 📊 Usage

Each section allows you to:
- Adjust parameters interactively with sliders
- Visualize data distributions with Plotly charts
- Download generated datasets as CSV files
- Experiment with different data characteristics

### Recommended Learning Path

1. **Start with Statistical Distributions** - Understand data shapes
2. **Linear vs Non-Linear** - See how model complexity needs vary
3. **Class Imbalance** - Learn why accuracy isn't enough
4. **Noise & Outliers** - Build robust models
5. **Time-Series** - Understand temporal data
6. **Categorical Data** - Practice encoding techniques

## 📁 Project Structure

```
ml-data-generator/
├── data_generator.py      # Main Streamlit application
├── pyproject.toml         # Project configuration and dependencies
├── uv.lock                # Dependency lockfile
├── README.md             # This file
├── LICENSE               # MIT License
└── .gitignore            # Git ignore patterns
```

## 🛠️ Technologies Used

- **Streamlit** - Interactive web application
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation
- **Plotly** - Interactive visualizations
- **SciPy** - Statistical distributions
- **Scikit-learn** - Machine learning utilities
- **Faker** - Realistic fake data generation

## 📚 Educational Use Cases

- **Testing ML Algorithms**: Generate datasets with known properties
- **Learning Data Preprocessing**: Practice scaling, encoding, handling missing data
- **Understanding Model Bias**: See how class imbalance affects performance
- **Experimenting with Hyperparameters**: Generate varied datasets to test model robustness
- **Teaching ML Concepts**: Visual demonstrations for students

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new data generation features
- Submit pull requests
- Improve documentation

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built to help the ML/AI community learn through practical, hands-on data generation exercises.

---

Made with ❤️ for the ML/AI community
