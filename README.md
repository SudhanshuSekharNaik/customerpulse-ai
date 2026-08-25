# CustomerPulse AI

### AI-Powered Customer Decision Intelligence Platform

CustomerPulse AI transforms raw customer behavioral and transactional data into actionable customer intelligence through **churn prediction, customer segmentation, lifecycle modeling, explainable AI, anomaly detection, next-best-action recommendations, and an AI-powered analyst**.

The platform is designed around a simple idea:

> **Don't just predict what a customer will do. Understand why, identify what is changing, and support the next decision.**

---

## 🚀 Live Demo

**CustomerPulse AI**

https://customerpulse-ai-ve3h.onrender.com

> The application is deployed on Render. Free-tier instances may require a short wake-up period after inactivity.

---

# 🎯 What CustomerPulse AI Does

Traditional customer analytics often stops at a dashboard or a single churn score.

CustomerPulse connects multiple analytical layers:

```text
Customer Events
       ↓
Data Validation
       ↓
Feature Engineering
       ↓
┌──────────────┬───────────────┬──────────────┐
│ Segmentation │ Churn Model   │ Anomaly      │
│              │               │ Detection    │
└──────┬───────┴───────┬───────┴──────┬───────┘
       │               │              │
       ▼               ▼              ▼
  Customer 360   SHAP Explainability  Alerts
       │
       ▼
Lifecycle Modeling
       │
       ▼
Next-Best Action
       │
       ▼
AI Analyst
       │
       ▼
Decision Dashboard
```

---

# 📊 Dataset

CustomerPulse currently works with an event-level e-commerce dataset containing transactional and behavioral customer activity.

### Dataset Profile

| Metric                  |                           Value |
| ----------------------- | ------------------------------: |
| Events                  |                      **12,000** |
| Unique Customers        |                       **1,299** |
| Unique Orders           |                      **12,000** |
| Products                |                          **22** |
| Categories              |                           **5** |
| Cities                  |                          **12** |
| Channels                |                           **4** |
| Payment Methods         |                           **5** |
| Date Range              | **Sep 16, 2025 – Mar 15, 2026** |
| Total Sales Amount      |                    **₹3.66 Cr** |
| Returns                 |                          **41** |
| Return Rate             |                       **0.34%** |
| Average Customer Rating |                    **4.86 / 5** |

### Event Distribution

| Event       |     Count |
| ----------- | --------: |
| View        | **7,813** |
| Add to Cart | **2,903** |
| Purchase    | **1,284** |

The event-level structure allows the system to derive customer-level behavioral features rather than relying only on static customer attributes.

---

# 🧠 Feature Engineering

Raw events are transformed into customer-level analytical features.

Examples include:

* Recency
* Purchase frequency
* Monetary value
* Average order value
* Purchase count
* Event activity
* View-to-cart behavior
* Cart-to-purchase behavior
* Return behavior
* Rating behavior
* Engagement signals
* Channel activity
* Product/category preferences

These features form the common analytical representation used across the ML pipeline.

---

# 🔮 Churn Prediction

CustomerPulse uses a **gradient-boosted classification model** to estimate individual customer churn risk.

The pipeline includes:

```text
Customer Events
      ↓
Customer-level Features
      ↓
Gradient Boosting Model
      ↓
Probability Calibration
      ↓
Individual Churn Probability
```

### Model Validation

The current documented model evaluation includes:

| Metric      |              Result |
| ----------- | ------------------: |
| PR-AUC      |          **0.7447** |
| Brier Score |          **0.3874** |
| Calibration | **Platt / Sigmoid** |
| Validation  |     **Out-of-Time** |

### Why PR-AUC?

Churn prediction is generally more useful to evaluate with precision-recall behavior than accuracy alone, particularly when the positive class is relatively less frequent.

### Why Out-of-Time Validation?

Customer behavior changes over time.

A random train/test split can allow temporal patterns from later observations to leak into training.

CustomerPulse instead uses a temporal validation strategy:

```text
Historical Customer Data
          │
          ├──────────────► Training Period
          │
          └──────────────► Future Holdout
                                │
                                ▼
                         Out-of-Time Evaluation
```

---

# 🔍 Explainable AI — TreeSHAP

CustomerPulse does not treat churn probability as a black-box output.

For individual predictions, **TreeSHAP** is used to identify the features contributing to the model's decision.

Example signals include:

* High recency
* Declining purchase frequency
* Reduced engagement
* Historical spending behavior
* Recent customer activity
* Return behavior

The resulting explanation can be surfaced through Customer 360 and the analytical interface.

```text
Customer Features
       ↓
Churn Model
       ↓
Prediction
       ↓
TreeSHAP
       ↓
Feature Contributions
       ↓
Human-readable Explanation
```

---

# 👥 Customer Segmentation

CustomerPulse applies **K-Means clustering** to identify behavioral customer groups.

The segmentation process evaluates multiple candidate cluster configurations using clustering quality metrics rather than assuming an arbitrary number of segments.

Typical segmentation signals include:

* Recency
* Frequency
* Monetary value
* Engagement
* Purchase behavior

The resulting segments can then be examined through the executive dashboard and Customer 360.

---

# 🔄 Customer Lifecycle Modeling

Customer behavior is not static.

CustomerPulse models movement between behavioral lifecycle states using **state-transition / Markov modeling**.

A simplified lifecycle can look like:

```text
NEW
 ↓
EXPLORING
 ↓
ENGAGED
 ↓
CONVERTING
 ↓
LOYAL
 ↓
DECLINING
 ↓
AT_RISK
 ↓
DORMANT
```

This provides a complementary perspective to churn prediction.

### Churn Prediction

> **How likely is this customer to churn?**

### Lifecycle Modeling

> **What behavioral state is this customer in, and where are they moving?**

---

# 🚨 Anomaly Radar

The Anomaly Radar identifies unusual changes in customer behavior.

Potential signals include:

* Sudden engagement decline
* Unusual purchase frequency
* Recency deterioration
* Spending deviations
* Abnormal behavioral changes

The objective is to detect changes that may require attention before they become visible in aggregate business metrics.

---

# 🎯 Next-Best Action

CustomerPulse converts customer intelligence into decision-support recommendations.

Recommendations can use:

* Churn risk
* Customer segment
* Lifecycle stage
* Customer value
* Recent behavior

Example actions include:

```text
WIN_BACK
RETENTION_OUTREACH
RE-ENGAGEMENT
LOYALTY_REWARD
VIP_ENROLLMENT
DISCOUNT
```

The recommendation system is intended as **decision support**.

It does not automatically execute customer-facing business actions.

> Recommendations should not be interpreted as experimentally validated causal treatment effects unless an explicit causal/uplift model is used.

---

# 🤖 AI Analyst Studio

CustomerPulse includes an AI-powered analyst that allows users to query customer intelligence using natural language.

Instead of manually navigating multiple analytical views, users can ask questions such as:

```text
Which customers have the highest churn risk?

What are the major drivers of churn?

Which customer segment is most valuable?

Why is this customer considered high risk?

Which customers have recently changed their behavior?
```

The analyst can interact with structured analytical tools to retrieve and analyze customer information.

### Agentic Workflow

```text
User Question
      ↓
AI Analyst
      ↓
Intent / Tool Selection
      ↓
Analytical Tool
      ↓
Customer / Model Data
      ↓
Analysis
      ↓
Natural-language Response
```

The execution flow is exposed through the interface to make the analytical process more transparent.

---

# 👤 Customer 360

Customer 360 combines the outputs of multiple analytical components into an individual customer profile.

```text
                    Customer 360
                         │
       ┌─────────────────┼─────────────────┐
       │                 │                 │
       ▼                 ▼                 ▼
   Behavioral         Churn Risk       Segment
    Profile
       │                 │                 │
       └─────────────────┼─────────────────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
          Lifecycle    SHAP       Anomaly
                       Factors     Signals
              │          │          │
              └──────────┼──────────┘
                         ▼
                  Next-Best Action
```

This provides a unified view of **who the customer is, what they are doing, what the model predicts, why it predicts it, and what action may be appropriate**.

---

# 📈 Executive Intelligence

The executive dashboard provides high-level visibility into:

* Customer population
* Revenue
* Active customers
* Churn exposure
* Customer segments
* Lifecycle states
* Behavioral anomalies
* High-risk customers
* Recommended actions

The dashboard is designed to support both:

**Executive → aggregate business intelligence**

and

**Analyst → customer-level investigation**

---

# 🧪 MLOps & Data Provenance

CustomerPulse includes engineering components for maintaining analytical traceability.

These include:

* Data validation
* Feature processing
* Dataset provenance
* Model metadata
* Model versioning
* Data-quality checks
* SHA-256 dataset fingerprinting
* Analytical audit information

Dataset fingerprinting helps establish which dataset version produced a particular analytical result.

---

# 🏗️ Architecture

```text
                         CSV
                          │
                          ▼
                  Data Validation
                          │
                          ▼
                  Feature Engineering
                          │
          ┌───────────────┼────────────────┐
          │               │                │
          ▼               ▼                ▼
      K-Means         Churn Model       Anomaly
    Segmentation      + Calibration     Detection
          │               │                │
          │               ▼                │
          │            TreeSHAP             │
          │               │                │
          └───────────────┼────────────────┘
                          ▼
                     Customer 360
                          │
          ┌───────────────┼────────────────┐
          │               │                │
          ▼               ▼                ▼
     Lifecycle       Next-Best         AI Analyst
      Modeling         Action             │
          │               │                │
          └───────────────┼────────────────┘
                          ▼
                  Decision Dashboard
```

---

# 🛠️ Technology Stack

## Machine Learning

* Python
* Scikit-learn
* Gradient Boosting
* K-Means
* TreeSHAP
* Probability Calibration
* Behavioral Anomaly Detection
* Markov / State-Transition Modeling

## AI

* LLM-powered analytical interface
* Tool calling
* Structured data analysis
* Agentic analytical workflow

## Backend

* Python
* REST APIs
* Modular ML services
* Data processing pipelines

## Frontend

* React
* Interactive analytics dashboard
* Customer intelligence views

## Deployment

* Docker
* Render
* Environment-based configuration

---

# 📁 Repository Structure

```text
customerpulse-ai/
│
├── ai/                  # AI Analyst and intelligent analysis
├── backend/app/         # Backend API and services
├── docs/                # Documentation
├── frontend/            # React frontend
├── ml/                  # Machine learning components
├── monitoring/          # Monitoring and quality components
├── pipelines/           # Data / ML pipelines
├── scripts/              # Utility scripts
├── tests/                # Automated tests
│
├── Dockerfile
├── docker-compose.yml
├── render.yaml
├── requirements.txt
└── README.md
```

---

# 🔬 End-to-End ML Pipeline

```text
Raw Event Data
      ↓
Data Validation
      ↓
Feature Engineering
      ↓
Customer Feature Matrix
      ↓
┌─────────────┬─────────────┬─────────────┬─────────────┐
│             │             │             │
▼             ▼             ▼             ▼
Segmentation  Churn         Lifecycle     Anomaly
              Prediction    Modeling      Detection
                  │
                  ▼
              Calibration
                  │
                  ▼
               TreeSHAP
                  │
└─────────────┴─────────────┴─────────────┘
                  ↓
             Customer 360
                  ↓
          Decision Intelligence
                  ↓
             AI Analyst
```

---

# 🔐 Design Principles

### Explainability

Predictions should provide interpretable evidence rather than exposing only a probability.

### Temporal Integrity

Model validation should respect the temporal nature of customer behavior.

### Data Provenance

Analytical results should be traceable to the underlying dataset and model version.

### Decision Support

The system recommends actions rather than silently executing business operations.

### Modular Architecture

Individual analytical components can evolve independently while contributing to a shared customer intelligence layer.

---

# ⚠️ Limitations

CustomerPulse AI is currently a **portfolio/production-style prototype**, rather than a fully hardened enterprise SaaS platform.

Important limitations include:

* Model performance depends on the underlying dataset.
* Churn probability distributions depend on customer behavior and feature distributions.
* Lifecycle transition estimates require sufficient temporal history.
* Next-best-action recommendations are decision-support outputs, not automatically validated causal effects.
* Dataset changes can produce different model metrics and customer risk distributions.
* Enterprise-scale multi-tenancy, advanced RBAC, and high-availability infrastructure can be extended further.
* AI Analyst responses depend on the quality and availability of the underlying analytical tools/data.

---

# 📌 Project Status

**Status: Active**

CustomerPulse AI is continuously evolving through:

* New datasets
* Model experiments
* Validation improvements
* Analytical modules
* Agentic AI capabilities
* MLOps improvements
* UI/UX improvements

---

# 🚀 Running Locally

### Clone

```bash
git clone https://github.com/SudhanshuSekharNaik/customerpulse-ai.git
cd customerpulse-ai
```

### Create virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure environment

Create the required `.env` configuration according to the project environment settings.

### Start the application

Run the backend and frontend using the project's development configuration.

---

# 🌐 Deployment

CustomerPulse AI is deployed using **Render** with containerized deployment support.

**Live Application:**

https://customerpulse-ai-ve3h.onrender.com

---

# 👨‍💻 Author

**Sudhanshu Sekhar Naik**

B.Tech — Information Technology

GitHub:

https://github.com/SudhanshuSekharNaik/customerpulse-ai

---

## Core Idea

> **A prediction is useful only when you can explain it, trust it, and connect it to a decision.**
