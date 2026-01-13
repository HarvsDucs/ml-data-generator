import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from faker import Faker

st.set_page_config(page_title="ML Data Generator", layout="wide")

fake = Faker()


def main():
    st.title("🤖 ML/AI Data Generator")
    st.markdown("Generate synthetic datasets to learn machine learning concepts")

    category = st.sidebar.selectbox(
        "Select Data Type",
        [
            "Statistical Distributions",
            "Linear vs Non-Linear",
            "Class Imbalance",
            "Noise & Outliers",
            "Time-Series",
            "Categorical Data",
        ],
    )

    if category == "Statistical Distributions":
        statistical_distributions()
    elif category == "Linear vs Non-Linear":
        linear_nonlinear()
    elif category == "Class Imbalance":
        class_imbalance()
    elif category == "Noise & Outliers":
        noise_outliers()
    elif category == "Time-Series":
        time_series()
    elif category == "Categorical Data":
        categorical_data()


def statistical_distributions():
    st.header("📊 Statistical Distributions")
    st.info(
        "Most ML models assume normal distribution. Learn how data shape affects model performance."
    )

    n_samples = st.slider("Number of samples", 100, 10000, 1000)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Normal (Gaussian)")
        mean = st.slider("Mean", -10, 10, 0, key="norm_mean")
        std = st.slider("Std Dev", 1, 10, 2, key="norm_std")
        normal_data = np.random.normal(mean, std, n_samples)

        fig = px.histogram(
            x=normal_data, nbins=50, title=f"Normal Distribution (μ={mean}, σ={std})"
        )
        st.plotly_chart(fig, use_container_width=True)

        df = pd.DataFrame({"value": normal_data, "type": "normal"})
        csv1 = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Normal Data", csv1, "normal_data.csv", "text/csv")

    with col2:
        st.subheader("Uniform Distribution")
        low = st.slider("Min Value", -10, 5, 0, key="uni_low")
        high = st.slider("Max Value", 5, 20, 10, key="uni_high")
        uniform_data = np.random.uniform(low, high, n_samples)

        fig = px.histogram(
            x=uniform_data, nbins=50, title=f"Uniform Distribution ({low} to {high})"
        )
        st.plotly_chart(fig, use_container_width=True)

        df = pd.DataFrame({"value": uniform_data, "type": "uniform"})
        csv2 = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Uniform Data", csv2, "uniform_data.csv", "text/csv"
        )

    with col3:
        st.subheader("Power Law (Long-tail)")
        alpha = st.slider("Alpha (shape)", 1.1, 5.0, 2.5, 0.1, key="plaw_alpha")
        power_data = (np.random.pareto(alpha, n_samples) + 1) * 1000

        fig = px.histogram(
            x=power_data, nbins=50, title=f"Power Law (α={alpha}) - Skewed Data"
        )
        st.plotly_chart(fig, use_container_width=True)

        df = pd.DataFrame({"value": power_data, "type": "power_law"})
        csv3 = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Power Law Data", csv3, "power_law_data.csv", "text/csv"
        )

    if st.button("Generate Combined Dataset"):
        combined = pd.DataFrame(
            {"normal": normal_data, "uniform": uniform_data, "power_law": power_data}
        )
        csv = combined.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download All Data", csv, "all_distributions.csv", "text/csv"
        )
        st.dataframe(combined.head(100))


def linear_nonlinear():
    st.header("📈 Linear vs Non-Linear Relationships")
    st.info(
        "Generate different relationship types to understand model complexity needs."
    )

    n_samples = st.slider("Number of samples", 100, 5000, 1000, key="ln_samples")
    x = np.random.uniform(-5, 5, n_samples)
    noise_level = st.slider("Noise Level", 0.0, 5.0, 1.0)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Linear")
        m = st.slider("Slope (m)", -5, 5, 2, key="linear_m")
        b = st.slider("Intercept (b)", -10, 10, 0, key="linear_b")
        noise = np.random.normal(0, noise_level, n_samples)
        y_linear = m * x + b + noise

        fig = px.scatter(x=x, y=y_linear, title=f"Linear: y = {m}x + {b}")
        st.plotly_chart(fig, use_container_width=True)

        df = pd.DataFrame({"x": x, "y": y_linear, "type": "linear"})
        csv1 = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Linear Data", csv1, "linear_data.csv", "text/csv")

    with col2:
        st.subheader("Polynomial")
        degree = st.selectbox("Degree", [2, 3, 4], key="poly_degree")
        a = st.slider("Coefficient", 0.1, 3.0, 1.0, key="poly_a")
        noise = np.random.normal(0, noise_level, n_samples)

        if degree == 2:
            y_poly = a * x**2 + noise
            title = f"Quadratic: y = {a}x²"
        elif degree == 3:
            y_poly = a * x**3 + noise
            title = f"Cubic: y = {a}x³"
        else:
            y_poly = a * x**4 + noise
            title = f"Quartic: y = {a}x⁴"

        fig = px.scatter(x=x, y=y_poly, title=title)
        st.plotly_chart(fig, use_container_width=True)

        df = pd.DataFrame({"x": x, "y": y_poly, "type": f"polynomial_{degree}"})
        csv2 = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Polynomial Data", csv2, "polynomial_data.csv", "text/csv"
        )

    with col3:
        st.subheader("Interaction Effects")
        x2 = np.random.uniform(-5, 5, n_samples)
        coef = st.slider("Interaction Coefficient", -3, 3, 1, key="inter_coef")
        noise = np.random.normal(0, noise_level, n_samples)
        y_interact = coef * x * x2 + noise

        df_interact = pd.DataFrame(
            {"x1": x, "x2": x2, "y": y_interact, "type": "interaction"}
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df_interact["x1"],
                y=df_interact["y"],
                mode="markers",
                name="Data points",
            )
        )
        fig.update_layout(
            title=f"Interaction: y = {coef} * x1 * x2",
            xaxis_title="x1",
            yaxis_title="y",
        )
        st.plotly_chart(fig, use_container_width=True)

        csv3 = df_interact.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Interaction Data", csv3, "interaction_data.csv", "text/csv"
        )


def class_imbalance():
    st.header("⚖️ Class Imbalance")
    st.warning(
        "Real-world scenarios like fraud detection have highly imbalanced classes. Accuracy is misleading here!"
    )

    n_samples = st.slider("Total samples", 1000, 100000, 10000, key="ci_samples")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Imbalanced Binary Classification")
        minority_ratio = st.slider("Minority Class Ratio (%)", 0.1, 50.0, 5.0)
        n_minority = int(n_samples * minority_ratio / 100)
        n_majority = n_samples - n_minority

        minority = np.random.multivariate_normal(
            [2, 2], [[1, 0.5], [0.5, 1]], n_minority
        )
        majority = np.random.multivariate_normal(
            [-2, -2], [[1, -0.3], [-0.3, 1]], n_majority
        )

        df = pd.DataFrame(
            np.vstack([minority, majority]), columns=["feature_1", "feature_2"]
        )
        df["label"] = [1] * n_minority + [0] * n_majority

        fig = px.scatter(
            df,
            x="feature_1",
            y="feature_2",
            color="label",
            title=f"Imbalanced: {minority_ratio}% minority class",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.metric("Majority Class", f"{n_majority} ({100 - minority_ratio}%)")
        st.metric("Minority Class", f"{n_minority} ({minority_ratio}%)")

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Imbalanced Data", csv, "imbalanced_data.csv", "text/csv"
        )

    with col2:
        st.subheader("SMOTE-like Oversampling")
        if st.button("Generate Synthetic Minority Samples"):
            k = st.slider("K neighbors for SMOTE", 1, 10, 5, key="smote_k")

            if len(minority) > k:
                synthetic = []
                for i in range(len(minority)):
                    distances = np.linalg.norm(minority - minority[i], axis=1)
                    neighbors = np.argsort(distances)[1 : k + 1]

                    for neighbor_idx in neighbors:
                        diff = minority[neighbor_idx] - minority[i]
                        gap = np.random.random()
                        synthetic_point = minority[i] + gap * diff
                        synthetic.append(synthetic_point)

                synthetic = np.array(synthetic)
                df_synthetic = pd.DataFrame(
                    synthetic, columns=["feature_1", "feature_2"]
                )
                df_synthetic["label"] = 1

                df_combined = pd.concat([df, df_synthetic], ignore_index=True)

                fig = px.scatter(
                    df_combined,
                    x="feature_1",
                    y="feature_2",
                    color="label",
                    title=f"After SMOTE: {len(synthetic)} synthetic samples added",
                )
                st.plotly_chart(fig, use_container_width=True)


def noise_outliers():
    st.header("🔊 Noise and Outliers")
    st.warning("Real-world data is dirty. Learn to make models robust!")

    n_samples = st.slider("Number of samples", 500, 10000, 2000, key="no_samples")
    x = np.random.uniform(-5, 5, n_samples)
    noise_level = st.slider("Gaussian Noise Level", 0.0, 5.0, 0.5, key="noise_level")

    tab1, tab2, tab3 = st.tabs(["Gaussian Noise", "Label Noise", "Outliers"])

    with tab1:
        st.subheader("Gaussian Noise")
        noise = np.random.normal(0, noise_level, n_samples)
        y = 2 * x + 1 + noise

        col1, col2 = st.columns(2)
        with col1:
            y_clean = 2 * x + 1
            fig = px.scatter(x=x, y=y_clean, title="Clean Data (No Noise)")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.scatter(x=x, y=y, title=f"Noisy Data (σ={noise_level})")
            st.plotly_chart(fig, use_container_width=True)

        df = pd.DataFrame({"x": x, "y": y, "y_clean": y_clean})
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Noisy Data", csv, "noisy_data.csv", "text/csv")

    with tab2:
        st.subheader("Label Noise")
        label_flip_rate = st.slider("Label Flip Rate (%)", 0, 50, 5)

        y_true = (x > 0).astype(int)
        n_flips = int(len(y_true) * label_flip_rate / 100)
        flip_indices = np.random.choice(len(y_true), n_flips, replace=False)
        y_noisy = y_true.copy()
        y_noisy[flip_indices] = 1 - y_noisy[flip_indices]

        col1, col2 = st.columns(2)

        with col1:
            df_clean = pd.DataFrame({"x": x, "label": y_true})
            fig = px.scatter(
                df_clean, x="x", y=[0] * len(x), color="label", title="Clean Labels"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            df_noisy = pd.DataFrame({"x": x, "label": y_noisy})
            fig = px.scatter(
                df_noisy,
                x="x",
                y=[0] * len(x),
                color="label",
                title=f"Noisy Labels ({label_flip_rate}% flipped)",
            )
            st.plotly_chart(fig, use_container_width=True)

        accuracy = (y_true == y_noisy).mean() * 100
        st.metric("Label Accuracy", f"{accuracy:.1f}%")
        st.warning(f"{n_flips} labels were flipped!")

        df = pd.DataFrame({"x": x, "y_true": y_true, "y_noisy": y_noisy})
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Label Noise Data", csv, "label_noise_data.csv", "text/csv"
        )

    with tab3:
        st.subheader("Outliers")
        n_outliers = st.slider("Number of Outliers", 1, 100, 10)
        outlier_distance = st.slider("Outlier Distance (σ)", 5, 20, 10)

        x_clean = np.random.normal(0, 1, n_samples)
        y_clean = 2 * x_clean + 1

        x_outliers = np.random.choice([-1, 1], n_outliers) * outlier_distance
        y_outliers = np.random.choice([-1, 1], n_outliers) * outlier_distance

        x_with_outliers = np.concatenate([x_clean, x_outliers])
        y_with_outliers = np.concatenate([y_clean, y_outliers])

        df = pd.DataFrame({"x": x_with_outliers, "y": y_with_outliers})
        df["is_outlier"] = [False] * n_samples + [True] * n_outliers

        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="is_outlier",
            title=f"Data with {n_outliers} Outliers (±{outlier_distance}σ)",
        )
        st.plotly_chart(fig, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Outlier Data", csv, "outlier_data.csv", "text/csv")


def time_series():
    st.header("⏰ Time-Series Variations")
    st.info(
        "Time-series data has temporal structure: trends, seasonality, and stationarity."
    )

    n_points = st.slider("Number of Time Points", 50, 1000, 365, key="ts_points")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Trend Only")
        trend_type = st.selectbox(
            "Trend Type", ["Linear", "Exponential"], key="trend_type"
        )
        slope = st.slider("Trend Slope", 0.01, 2.0, 0.5, key="trend_slope")

        t = np.arange(n_points)
        if trend_type == "Linear":
            trend = slope * t
        else:
            trend = np.exp(slope * t / 100)

        noise = np.random.normal(0, 0.1, n_points)
        data = trend + noise

        df = pd.DataFrame({"time": t, "value": data})
        fig = px.line(df, x="time", y="value", title=f"{trend_type} Trend")
        st.plotly_chart(fig, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Trend Data", csv, "trend_data.csv", "text/csv")

    with col2:
        st.subheader("Seasonality")
        period = st.slider("Period", 10, 100, 30, key="season_period")
        amplitude = st.slider("Amplitude", 0.1, 5.0, 1.0, key="season_amp")

        t = np.arange(n_points)
        seasonal = amplitude * np.sin(2 * np.pi * t / period)
        noise = np.random.normal(0, 0.1, n_points)
        data = seasonal + noise

        df = pd.DataFrame({"time": t, "value": data})
        fig = px.line(df, x="time", y="value", title=f"Seasonal (Period={period})")
        st.plotly_chart(fig, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Seasonal Data", csv, "seasonal_data.csv", "text/csv"
        )

    with col3:
        st.subheader("Trend + Seasonality")
        trend_slope = st.slider("Trend Slope", 0.01, 1.0, 0.2, key="ts_trend_slope")
        season_period = st.slider(
            "Seasonal Period", 10, 100, 52, key="ts_season_period"
        )
        season_amp = st.slider("Seasonal Amplitude", 0.1, 3.0, 1.0, key="ts_season_amp")

        t = np.arange(n_points)
        trend = trend_slope * t
        seasonal = season_amp * np.sin(2 * np.pi * t / season_period)
        noise = np.random.normal(0, 0.2, n_points)
        data = trend + seasonal + noise

        df = pd.DataFrame(
            {"time": t, "value": data, "trend": trend, "seasonal": seasonal}
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=df["time"], y=df["value"], name="Data", mode="lines")
        )
        fig.add_trace(
            go.Scatter(
                x=df["time"], y=df["trend"], name="Trend", line=dict(dash="dash")
            )
        )
        fig.update_layout(title="Trend + Seasonality + Noise")
        st.plotly_chart(fig, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Combined TS Data", csv, "combined_ts_data.csv", "text/csv"
        )

    st.subheader("Stationarity Test")
    st.info(
        "Non-stationary data has changing mean/variance over time. Models fail on this!"
    )

    if st.button("Generate Stationary vs Non-Stationary"):
        t = np.arange(n_points)

        stationary = np.random.normal(0, 1, n_points)

        non_stationary = np.cumsum(np.random.normal(0.1, 1, n_points))

        col1, col2 = st.columns(2)

        with col1:
            df_stat = pd.DataFrame({"time": t, "value": stationary})
            fig = px.line(
                df_stat, x="time", y="value", title="Stationary (Random Walk)"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            df_nonstat = pd.DataFrame({"time": t, "value": non_stationary})
            fig = px.line(
                df_nonstat, x="time", y="value", title="Non-Stationary (with Drift)"
            )
            st.plotly_chart(fig, use_container_width=True)


def categorical_data():
    st.header("🏷️ Categorical Data")
    st.info(
        "Use Faker to generate realistic categorical data for encoding practice (One-Hot, Label, Target)."
    )

    n_rows = st.slider("Number of Rows", 10, 10000, 1000, key="cat_rows")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Basic Categorical Features")
        n_categories = st.slider("Number of Categories", 2, 50, 5, key="n_cat")

        categories = [f"Category_{i}" for i in range(n_categories)]
        cat_feature = np.random.choice(
            categories, n_rows, p=np.random.dirichlet(np.ones(n_categories))
        )

        binary_feature = np.random.choice(["Yes", "No"], n_rows, p=[0.3, 0.7])

        ordinal_feature = np.random.choice(
            ["Low", "Medium", "High"], n_rows, p=[0.4, 0.4, 0.2]
        )

        df = pd.DataFrame(
            {
                "id": range(n_rows),
                "category": cat_feature,
                "binary": binary_feature,
                "ordinal": ordinal_feature,
            }
        )

        st.dataframe(df.head(50))

        st.write("Category Distribution:")
        st.write(df["category"].value_counts().to_frame().T)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Categorical Data", csv, "categorical_data.csv", "text/csv"
        )

    with col2:
        st.subheader("Realistic Data (Faker)")

        name_cardinality = st.slider(
            "High Cardinalinality (Names)", 10, 1000, 100, key="name_card"
        )
        country_cardinality = st.slider(
            "Country Cardinality", 5, 50, 20, key="country_card"
        )

        names = [fake.name() for _ in range(name_cardinality)]
        countries = [fake.country() for _ in range(country_cardinality)]

        data = {
            "id": range(n_rows),
            "name": np.random.choice(names, n_rows),
            "email": [fake.email() for _ in range(n_rows)],
            "country": np.random.choice(countries, n_rows),
            "job_title": np.random.choice(
                ["Engineer", "Data Scientist", "Manager", "Analyst"], n_rows
            ),
            "department": np.random.choice(
                ["Sales", "Engineering", "HR", "Marketing"], n_rows
            ),
            "salary": np.random.randint(50000, 150000, n_rows),
            "years_experience": np.random.randint(0, 20, n_rows),
        }

        df = pd.DataFrame(data)

        st.dataframe(df.head(50))

        st.write("Cardinality Summary:")
        st.write(df.nunique().to_frame().T)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Realistic Data", csv, "realistic_data.csv", "text/csv"
        )

    st.subheader("Encoding Practice")
    st.markdown("""
    **Encoding Techniques to Try:**
    - **One-Hot Encoding**: Use for low cardinality (<10 categories)
    - **Label Encoding**: Use for ordinal data
    - **Target Encoding**: Use for high cardinality (be careful with leakage!)
    - **Frequency Encoding**: Replace with count/frequency of each category
    
    **Privacy Note:** Never use real customer data for learning. Use Faker-generated data!
    """)


if __name__ == "__main__":
    main()
