<div align="center">

# 🎓 Student Insights Pro
### Data exploration · Machine learning · Evaluation · Predictions

A deployable data-science dashboard built with Python, Streamlit, Pandas, scikit-learn and Plotly.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white) ![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white) ![ML](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white) ![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)

</div>

## Features
- Responsive dark dashboard and synthetic demo data.
- Upload your own CSV and choose target/features.
- Data preview, missing values, duplicate counts, target distribution and correlation heatmap.
- ML pipeline with numeric/categorical imputation, one-hot encoding and Random Forest regression.
- Holdout evaluation with MAE, RMSE, R², actual-vs-predicted and residual plots.
- Interactive single-row prediction and CSV exports.
- Dockerfile and GitHub Actions test workflow.

## Project structure
```text
student_insights_pro/
├── app/main.py
├── src/pipeline.py
├── tests/test_pipeline.py
├── data/README.md
├── .github/workflows/tests.yml
├── requirements.txt
├── Dockerfile
└── README.md
```

## Run locally
Python 3.11+ recommended.

```bash
git clone https://github.com/Rakesh8993/student-insights-pro.git
cd student-insights-pro
python -m venv .venv
```

Activate environment:
- Windows PowerShell: `\.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Then:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app/main.py
```
Open the local URL printed by Streamlit, normally `http://localhost:8501`.

## Tests
```bash
pytest -q
```

## Docker
```bash
docker build -t student-insights-pro .
docker run --rm -p 8501:8501 student-insights-pro
```
Then open `http://localhost:8501`.

## Deploy to Streamlit Community Cloud
1. Push the project to GitHub.
2. Visit https://share.streamlit.io/ and create an app from your repository.
3. Set the main file path to `app/main.py`.
4. Deploy; dependencies are installed from `requirements.txt`.

## Uploading a dataset
Upload a CSV in the sidebar. Select a numeric target column and one or more feature columns. Missing feature values are handled inside the preprocessing pipeline. Ensure each row is one observation and do not include the target among the features. Keep private student data out of public deployments.

## Metrics
- **MAE:** average absolute error, in target units.
- **RMSE:** weights larger errors more heavily.
- **R²:** relative to a mean-prediction baseline; it may be negative.

Metric quality depends on the data, target scale, and baseline. Validate against representative held-out data before real-world use.

## Responsible use
The built-in demo dataset is synthetic, not real student data. Predictions are estimates and should not be the sole basis for decisions about students. Before production use, add authentication, data governance, monitoring, drift checks, stronger validation and model versioning.

## Author
**Rakesh Jagani** · [GitHub](https://github.com/Rakesh8993) · [Email](mailto:rkjagani7@gmail.com)
