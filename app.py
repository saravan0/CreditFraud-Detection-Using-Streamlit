import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import plotly.express as px
import plotly.graph_objects as go
import random

# Initialize session state variables
if "model_results" not in st.session_state:
    st.session_state.model_results = None

if "hyperparameters" not in st.session_state:
    st.session_state.hyperparameters = {
        "rf_n_estimators": 100,
        "rf_max_depth": 10,
        "rf_min_samples_split": 2,
        "dt_max_depth": 10,
        "dt_min_samples_split": 2,
        "dt_min_samples_leaf": 1,
        "lr_C": 1.0,
        "lr_max_iter": 100,
        "svm_C": 1.0,
        "svm_kernel": "rbf",
        "svm_gamma": "scale",
        "knn_n_neighbors": 5,
        "knn_weights": "uniform",
        "knn_algorithm": "auto",
    }


# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("creditCardFraud_28011964_120214.csv")
    df.rename(columns={"PAY_0": "PAY_1"}, inplace=True)
    return df


@st.cache_data
def prepare_data(df, sampling_method=None):
    x = df.drop("default payment next month", axis=1)
    y = df["default payment next month"]

    class_counts = y.value_counts()
    is_imbalanced = (class_counts.min() / class_counts.max()) < 0.8  # 80% threshold

    if is_imbalanced and sampling_method:
        if sampling_method == "Upsample":
            sampler = RandomOverSampler(random_state=42)
        elif sampling_method == "Downsample":
            sampler = RandomUnderSampler(random_state=42)
        x_resampled, y_resampled = sampler.fit_resample(x, y)  # type: ignore
    else:
        x_resampled, y_resampled = x, y

    X_train, X_test, y_train, y_test = train_test_split(
        x_resampled, y_resampled, test_size=0.3, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, is_imbalanced


# Train models
@st.cache_resource
def train_models(X_train, y_train, hyperparameters):
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=hyperparameters.get("rf_n_estimators", 100),
            max_depth=hyperparameters.get("rf_max_depth", None),
            min_samples_split=hyperparameters.get("rf_min_samples_split", 2),
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=hyperparameters.get("dt_max_depth", None),
            min_samples_split=hyperparameters.get("dt_min_samples_split", 2),
            min_samples_leaf=hyperparameters.get("dt_min_samples_leaf", 1),
            random_state=42,
        ),
        "Logistic Regression": LogisticRegression(
            C=hyperparameters.get("lr_C", 1.0),
            max_iter=hyperparameters.get("lr_max_iter", 100),
            random_state=42,
        ),
        "Support Vector Machine": SVC(
            C=hyperparameters.get("svm_C", 1.0),
            kernel=hyperparameters.get("svm_kernel", "rbf"),
            gamma=hyperparameters.get("svm_gamma", "scale"),
            random_state=42,
        ),
        "K-Nearest Neighbour": KNeighborsClassifier(
            n_neighbors=hyperparameters.get("knn_n_neighbors", 5),
            weights=hyperparameters.get("knn_weights", "uniform"),
            algorithm=hyperparameters.get("knn_algorithm", "auto"),
        ),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

    return models


# Load data and prepare it
df = load_data()
sampling_method = None
X_train_scaled, X_test_scaled, y_train, y_test, _ = prepare_data(
    df,
    sampling_method if sampling_method != "None" else None,
)

# Streamlit app
st.title(":orange[CreditFraud💳: Streamlit powered Multi-Model Predictive Analysis Platform for Credit Card Default Detection]")

# Create navigation radio buttons 
nav=st.sidebar.radio("Navigation",  ["Home", "Explore the Dataset", "EDA", "ML Model", "Future"])


# Create a button for the Easter egg
with st.sidebar:
    st.write(":red[Easter Egg!]")
    easter_egg_button = st.button("Click Me!")

    # Define the Easter egg functionality
    if easter_egg_button:
        st.info("Click again to generate a random **:red[quote]** or **:red[FunFact]** or **:red[Joke]**")
        # Randomly select an Easter egg type
        easter_egg_type = random.choice(["Fun Fact", "A Joke", "Quote"])

        if easter_egg_type == "Fun Fact":
            # Display a randomly generated fun fact about machine learning or AI
            fun_facts = [
                "The first neural network was created in 1943 by Warren McCulloch and Walter Pitts.",
                "The term 'Artificial Intelligence' was coined in 1956 by John McCarthy.",
                "The first chatbot, ELIZA, was developed in 1966 by Joseph Weizenbaum.",
                "Google's AlphaGo AI defeated a human world champion in Go in 2016.",
                "The first self-driving car was developed in 1986 by Ernst Dickmanns."
            ]
            st.write("🤔 Did you know that...")
            st.write(random.choice(fun_facts))

        elif easter_egg_type == "A Joke":
            # Display a randomly generated joke
            jokes = [
                "Why don't scientists trust atoms? Because they make up everything.",
                "Why don't eggs tell jokes? They'd crack each other up.",
                "Why did the tomato turn red? Because it saw the salad dressing!",
                "What do you call a fake noodle? An impasta.",
                "Why did the scarecrow win an award? Because he was outstanding in his field."
            ]
            st.write("😄 Here's a joke for you:")
            st.write(random.choice(jokes))

        elif easter_egg_type == "Quote":
            # Display a randomly generated quote by an Indian, excluding Mahatma Gandhi and Nehru
            indian_quotes = [
                "The biggest risk is not taking any risk. - N. R. Narayana Murthy",
                "Innovation is the specific instrument of entrepreneurship. - Kiran Mazumdar-Shaw",
                "The only limit to our realization of tomorrow will be our doubts of today. - A. P. J. Abdul Kalam",
                "Success is not about how much money you make, but about the difference you make in people's lives. - Shiv Nadar",
                "The biggest adventure you can take is to live the life of your dreams. - Azim Premji"
            ]
            st.write("💡 Here's a quote for you:")
            st.write(random.choice(indian_quotes))

if nav=="Home":
    st.text('''
    __  __      _ _        __   __              __ 
    | | | | ___| | | ___   | |  | |___  ___ _ __| |
    | |_| |/ _ \ | |/ _ \  | |  | / __|/ _ \ '__| |
    |  _  |  __/ | | (_) | | |__| \__ \  __/ |  |_|
    |_| |_|\___|_|_|\___/  |_|__|_|___/\___|_|  (_)
    ''')
    st.markdown("""
    This Streamlit-based web application combines **data analysis**, **exploratory data analysis (EDA)**, and **machine learning models** to predict credit card fraud.

    ## 🔍 Key Features

    - Interactive data exploration and visualization
    - Multiple machine learning models with customizable hyperparameters
    - Handling of imbalanced datasets
    - Comprehensive model evaluation metrics
    - Educational components explaining various concepts and metrics

    ## 📊 Application Sections

    ### 1. Data Loading and Preprocessing
    - Loads credit card data from a CSV file
    - Performs basic data preprocessing, including renaming columns

    ### 2. Exploratory Data Analysis (EDA)
    Provides various visualizations to explore the dataset:
    - Correlation heatmap
    - Histograms for credit limit balance and age distribution
    - Box plots for credit limit by gender and age distribution by default status
    - Pie charts for gender distribution and default payment
    - Bar chart for education levels

    ### 3. Machine Learning Models
    """)

    st.info("""
    Implements several classification models:
    - Random Forest
    - Decision Tree
    - Logistic Regression
    - Support Vector Machine
    - K-Nearest Neighbors
    """)

    st.markdown("""
    - Allows users to select a model and adjust hyperparameters
    - Handles data imbalance through upsampling or downsampling options

    ### 4. Model Evaluation
    - Displays classification report with metrics like precision, recall, and F1-score
    - Shows confusion matrix and related metrics (TP, TN, FP, FN, TPR, FPR)
    - Plots ROC-AUC curve for applicable models

    ### 5. User Interface
    - Utilizes Streamlit for an interactive web interface
    - Includes navigation options for different sections of the application
    - Provides explanations for various metrics and visualizations
    """)

    st.success("This project serves as a comprehensive tool for analyzing credit card fraud, allowing users to explore the data, train different models, and evaluate their performance in predicting fraudulent transactions.")

    # Interactive element
    if st.button("Learn More About Credit Card Fraud"):
        st.write("Credit card fraud is a significant issue in the financial sector. Machine learning models can help detect fraudulent transactions by identifying patterns and anomalies in transaction data.")
        # Colorful metrics example
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", "85%", "2%")
        col2.metric("Precision", "89%", "-1%")
        col3.metric("Recall", "90%", "3%")

elif nav=="Explore the Dataset":
        
    # EDA starts
    show_button = st.button("👀:red[View file]")

    if df is not None:
        # Read the file
        st.success("Dataset has been loaded")

        # Option to view the file in a popup window
        if show_button:
            with st.expander("Contents of the file ", expanded=True):
                st.write(df)
                st.write(f"**:red[Dataset shape: {df.shape}]**")

        columns = df.columns
        with st.expander("The variables are ", expanded=False):
            st.dataframe(columns, hide_index=True)  # type: ignore

        target = df.columns[-1]
        with st.expander("The target variable is ", expanded=False):
            st.write(target)

        if st.checkbox("Show Data Overview"):
            st.write("**Brief description of the data**", df.describe())
            st.write("**Missing values per column:**", df.isna().sum())

        if st.checkbox("Show Variable Descriptions"):
            st.write("**Here are the descriptions of the variables:**")

            # Dictionary containing variable descriptions
            variable_descriptions = {
                "LIMIT_BAL": "This feature represents the credit limit assigned to the individual's credit card. it indicates the maximum amount of credit the person can utilize",
                "SEX": "SEX denotes the gender of the credit card holder. while gender doesnt directly impact the credit default detection, but it might be considered as a demographic factor that might have some influence on creditworthiness",
                "EDUCATION": "EDUCATION indicates the education ackground of the credit card holder. it provides the insights into the person's level of education which might indirectly correlate with their financial stability and ability to manage the credit 1: Graduate School (Postgraduate degree, e.g., Master's or PhD), 2: University (Undergraduate degree, e.g., Bachelor's), 3: High School, 4: Others (could include associate degrees, vocational training, or other non-university education), 5: Unknown (sometimes used for missing or unknown education levels), 6: Unknown (another possible category for unknown or not applicable education levels)",
                "MARRIAGE": "Marital status (1 = married, 2 = single, 3 = others).",
                "AGE": "AGE denotes the age of the credit card holder which might be a important factor in accessing the creditworthiness as it often correlates with the financial stability",
                "default payment next month": "this is the target variable that indicate the credit card holder defaulted on their payment in the following month (1 for default 0 for no default) this is the variable that the credit card fault detection model aims to predict ",
            }
            # Common descriptions for grouped columns
            pay_amt_description = "Amount of previous payments of past 6 months"
            bill_amt_description = "Amount of bill statement of past 6 months "
            pay_status_description = (
                "Repayment status (0 = paid in full, -1 = paid one month delay, etc"
            )

            # Add descriptions for PAY_AMT, BILL_AMT, and PAY_ columns dynamically
            for i in range(1, 7):
                variable_descriptions[f"PAY_AMT{i}"] = pay_amt_description
                variable_descriptions[f"BILL_AMT{i}"] = bill_amt_description
                variable_descriptions[f"PAY_{i}"] = pay_status_description
            for column in df.columns:
                description = variable_descriptions.get(column, "No description available.")
                st.write(f"**{column}**: {description}")
elif nav=="EDA":
    # Title of the Streamlit app
    st.title("Exploratory Data Analysis (EDA) Visualizations")

    # Display pairplot
    # Display pairplot
    ## st.subheader("Pairplot")
    ## pairplot_fig = sns.pairplot(df)
    ## st.pyplot(pairplot_fig.figure)

    # Display heatmap of correlations
    st.subheader("Heatmap of Correlations")
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(df.corr(), annot=True, cmap="RdBu", ax=ax)
    plt.tight_layout()
    st.pyplot(fig)

    if st.toggle("**:red[Heatmap Insights]📌**"):
        st.write(
            ":orange[The correlation heatmap shows that some of the features are related to each other, meaning they aren't completely independent. For example, if a customer missed a payment one month, they're likely to miss payments in the following months too, which is why we see a correlation. Similarly, if a customer couldn't pay a bill, the amount due usually stayed the same, and if they could pay, the amount decreased.We usually remove columns that provide the same information to avoid redundancy. However, in this case, removing columns would mean losing important information about payment history and bill amounts. So, even though there is a correlation between the columns, we should keep them]"
        )

    st.subheader("Correlation Matrix")
    selected_features = [
        "LIMIT_BAL",
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "AGE",
        "default payment next month",
    ]
    correlation_matrix = df[selected_features].corr()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        ax=ax,
    )
    plt.title("Correlation Heatmap of Selected Features")
    plt.tight_layout()
    st.pyplot(fig)

    
    # Display histogram of LIMIT_BAL 
    fig=px.histogram(df, x="LIMIT_BAL", nbins=30, marginal="violin")
    # Update the marker color using a color sequence
    fig.update_traces(marker_color="tomato")
    fig.update_layout(title="Distribution of Credit Limit Balance", xaxis_title="Limit Balance", yaxis_title="Count")
    st.plotly_chart(fig)
    # Display histogram of AGE
    fig = px.histogram(df, x="AGE", nbins=50, marginal="box")
    fig.update_traces(marker_color="magenta")
    fig.update_layout(
        title="Distribution of Age", xaxis_title="Age", yaxis_title="Count"
    )
    st.plotly_chart(fig)

    if st.toggle("**:red[Histogram Insights]📌**"):
        st.write(
            ":orange[The histograms illustrate the frequency distribution of customer ages and credit limits, revealing the most common ranges for each variable]"
        )

    # Display boxplot of LIMIT_BAL by SEX
    fig = px.box(df, x="SEX", y="LIMIT_BAL", color="SEX", points="all")
    fig.update_layout(
        title="Credit Limit Balance by Gender",
        xaxis_title="Gender (1-Female)(2-Male)",
        yaxis_title="Credit Limit Balance",
    )
    st.plotly_chart(fig)

    if st.toggle("**:red[BoxPlot of LimitBalance insights]📌**"):
        st.write(
            ":orange[The median credit limit appears slightly higher for group 2 (likely females). Both groups have similar interquartile ranges (box sizes). There are several outliers in both groups, with very high credit limits. The overall distribution of credit limits seems fairly similar between the two groups]"
        )

    # Display boxplot of AGE by Default Status
    fig = px.box(
        df,
        x="default payment next month",
        y="AGE",
        color="default payment next month",
        points="all",
    )
    fig.update_layout(
        title="Age Distribution by Default Status",
        xaxis_title="Default Status (0-Dont default)(1-Default)",
        yaxis_title="Age",
    )
    st.plotly_chart(fig)

    if st.toggle("**:red[BoxPlot of Age distribution insights]📌**"):
        st.write(
            ":orange[The median age for those who defaulted (1) appears slightly higher than for those who didn't (0). The interquartile range for the default group (1) is larger, suggesting more age variability. Both groups have outliers at the upper end, representing older individuals. The overall age distribution for non-defaulters (0) seems to be slightly lower and more compact]"
        )

    # Display countplot of customers by gender
    label_mapping = {
        1: "Female",
        2: "Male",
    }  
    df["SEX_Label"]=df["SEX"].map(label_mapping)
    sex_counts=df["SEX_Label"].value_counts()
    fig = px.pie(
    data_frame=df,
    names=sex_counts.index,
    values=sex_counts.values,
    title="Pie chart of gender distribution",
    hole=0.1,
    color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(width=400, height=400, legend_title="Gender")

    st.plotly_chart(fig, use_container_width=True)
    
    # Display bar chart of customers by education level
    fig = px.bar(
    x=df["EDUCATION"].value_counts().index,  # Use the index of value_counts as the x-axis
    y=df["EDUCATION"].value_counts().values,  # Use the values of value_counts as the y-axis
    title="Count of Customers by Education Level",
    )
    # Add x and y labels
    fig.update_layout(
        xaxis_title="Education Level",
        yaxis_title="Count of Customers",
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Display piechart of default payment next month
    label_mapping = {
        0: "Do not default",
        1: "Default",
    }  # Mapping numeric labels to descriptive labels
    labels = df["default payment next month"].map(label_mapping).value_counts().index
    values = df["default payment next month"].value_counts().values

    fig = px.pie(names=labels, values=values, 
             title="Piechart: Default Payment Next Month", 
             hole=0.1,  # Create a donut chart
             color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig, use_container_width=True)
    
    
elif nav=="ML Model":

    st.subheader(":red[Make sure the dataset is balanced before running the ML model]")
    y = df["default payment next month"]
    class_counts = y.value_counts()
    is_imbalanced = (class_counts.min() / class_counts.max()) < 0.8  # 80% threshold

    if st.toggle("**:red[🚨🚨🚨Check Here🚨🚨🚨]**"):  # type: ignore
        if is_imbalanced:
            st.warning("The dataset is imbalanced. Consider using a sampling method.")
            sampling_method = st.selectbox(
                "Choose a sampling method", ["None", "Upsample", "Downsample"]
            )
        else:
            sampling_method = "None"

    # # Prepare data with selected sampling method
    # if "sampling_method" in st.session_state:
    #     sampling_method=st.session_state.sampling_method
        X_train_scaled, X_test_scaled, y_train, y_test, _ = prepare_data(
        df, sampling_method if sampling_method != "None" else None
    )
    # Old Class Distribution - Pie Chart
    st.subheader("Old Class Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    y_counts = y.value_counts()
    ax.pie(
        y_counts,
        labels=y_counts.index,  # type: ignore
        autopct="%1.1f%%",
        startangle=90,
        colors=["red", "blue"],
        explode=[0, 0.1],
        frame=True,
        rotatelabels=True,
        normalize=True,
    )
    ax.set_title("Distribution of Default vs Non-Default in Training Data")
    st.pyplot(fig)

    # Display information about sampling
    if sampling_method != "None":
        st.info(f"Data has been balanced using {sampling_method}.")
        st.write("New class distribution:")
        st.write(y_train.value_counts())

    if sampling_method:
        # New Class Distribution - Pie Chart
        st.subheader("New Class Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        y_train_counts = y_train.value_counts()
        ax.pie(
            y_train_counts,
            labels=y_train_counts.index,
            autopct="%1.1f%%",
            startangle=45,
            colors=["red", "blue"],
            explode=[0, 0.1],
            frame=True,
            rotatelabels=True,
        )
        ax.set_title("New Distribution of Default vs Non-Default in Training Data")
        st.pyplot(fig)
    # Machine Learning Model
    st.header(":orange[Select a Machine Learning Model]🤖")

    model_choice = st.selectbox(
        "**Choose a Model**",
        list(
            train_models(
                X_train_scaled, y_train, st.session_state.hyperparameters
            ).keys()
        ),
    )

    st.subheader("Set Hyperparameters")

    if model_choice == "Random Forest":
        st.session_state.hyperparameters["rf_n_estimators"] = st.slider(
            "Number of trees",
            10,
            200,
            st.session_state.hyperparameters["rf_n_estimators"],
        )
        st.session_state.hyperparameters["rf_max_depth"] = st.slider(
            "Maximum depth", 1, 20, st.session_state.hyperparameters["rf_max_depth"]
        )
        st.session_state.hyperparameters["rf_min_samples_split"] = st.slider(
            "Minimum samples to split",
            2,
            10,
            st.session_state.hyperparameters["rf_min_samples_split"],
        )

    elif model_choice == "Decision Tree":
        st.session_state.hyperparameters["dt_max_depth"] = st.slider(
            "Maximum depth", 1, 20, st.session_state.hyperparameters["dt_max_depth"]
        )
        st.session_state.hyperparameters["dt_min_samples_split"] = st.slider(
            "Minimum samples to split",
            2,
            10,
            st.session_state.hyperparameters["dt_min_samples_split"],
        )
        st.session_state.hyperparameters["dt_min_samples_leaf"] = st.slider(
            "Minimum samples in leaf",
            1,
            10,
            st.session_state.hyperparameters["dt_min_samples_leaf"],
        )

    elif model_choice == "Logistic Regression":
        st.session_state.hyperparameters["lr_C"] = st.slider(
            "Inverse of regularization strength",
            0.01,
            10.0,
            st.session_state.hyperparameters["lr_C"],
        )
        st.session_state.hyperparameters["lr_max_iter"] = st.slider(
            "Maximum iterations",
            100,
            1000,
            st.session_state.hyperparameters["lr_max_iter"],
        )

    elif model_choice == "Support Vector Machine":
        st.session_state.hyperparameters["svm_C"] = st.slider(
            "Regularization parameter",
            0.01,
            10.0,
            st.session_state.hyperparameters["svm_C"],
        )
        st.session_state.hyperparameters["svm_kernel"] = st.selectbox(
            "Kernel",
            ["rbf", "linear", "poly"],
            index=["rbf", "linear", "poly"].index(
                st.session_state.hyperparameters["svm_kernel"]
            ),
        )
        st.session_state.hyperparameters["svm_gamma"] = st.selectbox(
            "Kernel coefficient",
            ["scale", "auto"],
            index=["scale", "auto"].index(
                st.session_state.hyperparameters["svm_gamma"]
            ),
        )

    elif model_choice == "K-Nearest Neighbour":
        st.session_state.hyperparameters["knn_n_neighbors"] = st.slider(
            "Number of neighbors",
            1,
            20,
            st.session_state.hyperparameters["knn_n_neighbors"],
        )
        st.session_state.hyperparameters["knn_weights"] = st.selectbox(
            "Weight function",
            ["uniform", "distance"],
            index=["uniform", "distance"].index(
                st.session_state.hyperparameters["knn_weights"]
            ),
        )
        st.session_state.hyperparameters["knn_algorithm"] = st.selectbox(
            "Algorithm",
            ["auto", "ball_tree", "kd_tree", "brute"],
            index=["auto", "ball_tree", "kd_tree", "brute"].index(
                st.session_state.hyperparameters["knn_algorithm"]
            ),
        )

    if st.button("**:green[Run Model]**"):
        # Retrain the model with new hyperparameters
        trained_models = train_models(
            X_train_scaled, y_train, st.session_state.hyperparameters
        )
        model = trained_models[model_choice]  # type: ignore
        y_pred = model.predict(X_test_scaled)

        # Store results in session state
        st.session_state.model_results = {
            "y_test": y_test,
            "y_pred": y_pred,
            "model": model,
            "X_test_scaled": X_test_scaled,
        }

    # After the button, check if results exist and display them
    if st.session_state.model_results is not None:
        y_test = st.session_state.model_results["y_test"]
        y_pred = st.session_state.model_results["y_pred"]
        model = st.session_state.model_results["model"]
        X_test_scaled = st.session_state.model_results["X_test_scaled"]

        st.write("**:violet[Classification Report:]📊**")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.write(pd.DataFrame(report).transpose())

        st.write("**:violet[Confusion Matrix:]**")
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Negative", "Actual Positive"],
            columns=["Predicted Negative", "Predicted Positive"],
        )
        st.write(cm_df)

        # Extract confusion matrix terms
        tn, fp, fn, tp = cm.ravel()
        st.write(f"True Positives (TP): {tp}")
        st.write(f"True Negatives (TN): {tn}")
        st.write(f"False Positives (FP): {fp}")
        st.write(f"False Negatives (FN): {fn}")
        st.write(f"True positive rate (TPR): {tp/(tp+fn)}")
        st.write(f"False positive rate (FPR): {fp/(fp+tn)}")

        with st.expander(
            ":orange[What do those metrics mean? Click here to find out (No google required):smiley:]"
        ):
            st.write(
                ":red[Accuracy]: The overall proportion of correct predictions (both positive and negative)"
            )
            st.write(
                ":red[Precision]: It measures how many of the positive predictions made by a model were actually correct."
            )
            st.write(
                ":red[Recall]: The proportion of actual positive cases correctly identified"
            )
            st.write(
                ":red[F1 Score]: A harmonic mean of precision and recall, providing a balance between the two"
            )
            st.write(
                ":red[True Positive(TP)]: In simple words TP is when a model correctly predicts a positive class✅✅"
            )
            st.write(
                ":red[True Negative(TN)]: In simple words TN is when model correctly predicts a negative class❌❌"
            )
            st.write(
                ":red[False Positive(FP)]: In simple words FP also called as **Type1 error**, is when a model incorrectly predicts a positive class❌✅"
            )
            st.write(
                ":red[False Negative(FN)]: In simple words FN also called as **Type2 error**, is when a model incorrectly predicts a negative class✅❌"
            )
            st.write(
                ":red[True Positive rate(TPR)]: TPR, also known as Sensitivity or Recall, is the proportion of actual positive cases that were correctly predicted as positive"
            )
            st.write(
                ":red[False Positive rate(FPR)]: FPR is the proportion of actual negative cases that were incorrectly predicted as positive"
            )

        # ROC AUC curve
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test_scaled)
        else:
            st.write(
                "This model doesn't support probability predictions or decision function. ROC curve cannot be plotted."
            )

        if "y_prob" in locals():
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=fpr, y=tpr, mode="lines", name=f"ROC curve (area = {roc_auc:.2f})"
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Baseline",
                    line=dict(dash="dash"),
                )
            )
            fig.update_layout(
                title="Receiver Operating Characteristic",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                width=700,
                height=500,
            )
            st.plotly_chart(fig)

        if st.toggle(
            ":red[**What's the graph trying to tell us? Click here to find out**]"
        ):
            st.write(
                "The ROC-AUC curve is a graphical representation that helps us understand how well a classification model performs. The AUC (Area Under the Curve) measures the model's ability to distinguish between positive and negative classes. A higher AUC means the model is better at distinguishing between the two classes."
            )
elif nav=="Future":
    st.info('''
            Things under development 
            - Develop a flexible web application for dataset ingestion and streamlined execution of machine learning project lifecycle stages.
            - Currently, the Python code requires a full re-run with each modification in the Streamlit app, which is inefficient. To address this, I am  planning to implement a multipage architecture, dividing the ML model building process into distinct stages, each on a separate page.
            - Implement diverse strategies for managing null values, encoding categorical data, and removal of sampling techniques(employed in this project) 
            - Integrate all available hyperparameters defined in the scikit-learn documentation, enabling users to customize and select optimal parameter values.
            - Introducing PandasProfiling to facilitate advanced exploratory data analysis capabilities.
            - Introducing Artificial Neural Networks(MultiLayer Perceptron)
            '''
        
    )
