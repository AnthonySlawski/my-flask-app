from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, classification_report
import plotly.graph_objects as go
import plotly.io as pio
import os

# Create a Flask app
app = Flask(__name__)

# Define categories for the diagnosis tool
categories = [
    "Block Design", "Similarities", "Matrix Reasoning", "Digit Span", "Coding",
    "Vocabulary", "Figure Weights", "Visual Puzzles", "Picture Span", "Symbol Search",
    "Information", "Picture Concepts", "Letter-Number Sequencing", "Cancellation",
    "Comprehension", "Arithmetic", "Letter Sound Knowledge", "Early Word Recognition",
    "Sound Deletion", "Sound Isolation", "Single Word Reading", "Reading Accuracy",
    "Reading Rate", "Reading Comprehension", "Fluency"
]

# Labels for dyslexia prediction
dyslexia_labels = {0: 'No Dyslexia', 1: 'Mild Dyslexia', 2: 'Severe Dyslexia'}

# Attribute descriptions
attribute_descriptions = {
    'Block Design': {'positive': 'Strong spatial visualization and problem-solving skills.',
                     'negative': 'Challenges with spatial visualization and problem-solving.'},
    'Similarities': {'positive': 'Strong skills in verbal reasoning and forming concepts.',
                     'negative': 'Challenges in verbal reasoning and forming concepts.'},
    'Matrix Reasoning': {'positive': 'Strong visual-spatial reasoning and fluid intelligence.',
                         'negative': 'Difficulty with visual-spatial reasoning and fluid intelligence.'},
    'Digit Span': {'positive': 'Strong attention, concentration, and mental control.',
                   'negative': 'Difficulty with attention, concentration, and mental control.'},
    'Coding': {'positive': 'Fast processing speed and good short-term visual memory.',
               'negative': 'Slow processing speed and poor short-term visual memory.'},
    'Vocabulary': {'positive': 'Rich vocabulary and strong word knowledge.',
                   'negative': 'Limited vocabulary and struggles with word knowledge.'},
    'Figure Weights': {'positive': 'Good quantitative reasoning and visual-perceptual skills.',
                       'negative': 'Challenges in quantitative reasoning and visual-perceptual skills.'},
    'Visual Puzzles': {'positive': 'Excellent nonverbal reasoning and problem-solving abilities.',
                       'negative': 'Struggles with nonverbal reasoning and problem-solving.'},
    'Picture Span': {'positive': 'Strong visual working memory.',
                     'negative': 'Weak visual working memory.'},
    'Symbol Search': {'positive': 'Strong processing speed and visual-motor coordination.',
                      'negative': 'Challenges with processing speed and visual-motor coordination.'},
    'Information': {'positive': 'Strong general knowledge and understanding of world facts.',
                    'negative': 'Limited knowledge of general information.'},
    'Picture Concepts': {'positive': 'Strong abstract reasoning and categorization skills.',
                         'negative': 'Difficulty with abstract reasoning and categorization.'},
    'Letter-Number Sequencing': {'positive': 'Strong working memory and adept at sequencing tasks.',
                                 'negative': 'Challenges with working memory and sequencing tasks.'},
    'Cancellation': {'positive': 'Quick visual recognition and focused attention.',
                     'negative': 'Slow visual recognition and scattered attention.'},
    'Comprehension': {'positive': 'Good understanding of social situations and common sense.',
                      'negative': 'Difficulty understanding social situations and common sense.'},
    'Arithmetic': {'positive': 'Strong calculation skills and numerical operations.',
                   'negative': 'Difficulty with calculations and numerical operations.'},
    'Letter Sound Knowledge': {'positive': 'Strong phonological skills and understanding of letter-sound relationships.',
                               'negative': 'Struggles with phonological processing and letter-sound connections.'},
    'Early Word Recognition': {'positive': 'Strong early reading skills and quick recognition of common words.',
                               'negative': 'Challenges with early reading and word recognition.'},
    'Sound Deletion': {'positive': 'Strong phonemic manipulation skills.',
                       'negative': 'Difficulty with phonemic manipulation tasks.'},
    'Sound Isolation': {'positive': 'Excellent at identifying sounds within words.',
                        'negative': 'Struggles with sound identification within words.'},
    'Single Word Reading': {'positive': 'Proficient at reading single words quickly and accurately.',
                            'negative': 'Difficulty reading single words accurately.'},
    'Reading Accuracy': {'positive': 'High level of reading accuracy.',
                         'negative': 'Issues with reading words accurately.'},
    'Reading Rate': {'positive': 'Fast reading speed.',
                     'negative': 'Slow reading speed.'},
    'Reading Comprehension': {'positive': 'Good understanding of written text.',
                              'negative': 'Difficulty understanding written text.'},
    'Fluency': {'positive': 'Smooth, fast, and expressive reading.',
                'negative': 'Halting, slow, and monotone reading.'},
}

# Ensure the CSV file is accessible
file_path = os.path.join(os.path.dirname(__file__), "MASTER_DATA.csv")

def load_data(file_path):
    # Load the original CSV file and remove the specified columns
    Data = pd.read_csv(file_path).drop(columns=["Subtest"])
    # Replace 'No Dyslexia' with 0, 'Mild' with 1, and 'Severe' with 2
    Data.replace({'No Dyslexia': 0, 'Mild': 1, 'Severe': 2}, inplace=True)
    # Separate the features (X) and the target (y)
    X = Data.iloc[0:25].values.T.astype(float)  # Transpose X to ensure rows are samples and columns are features
    y = Data.iloc[25].values.astype(int)  # Ensure y is an array of integers
    return X, y

def create_pipeline():
    # Define the pipeline with scaling, normalization, PCA, and the classifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),       # Standardize features by removing the mean and scaling to unit variance
        ('normalizer', Normalizer()),       # Normalize samples individually to unit norm
        ('pca', PCA()),                     # PCA for dimensionality reduction
        ('classifier', RandomForestClassifier(random_state=42))  # RandomForest classifier
    ])
    return pipeline

def train_model(X, y):
    # Split the data into training and testing sets using stratified split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    pipeline = create_pipeline()

    # Define the parameter grid
    param_grid = {
        'pca__n_components': [2, 3, 4],
        'classifier__n_estimators': [10, 50, 100]
    }

    # Create the GridSearchCV object with recall as the scoring metric and stratified k-fold cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=StratifiedKFold(n_splits=5), scoring='recall_macro', verbose=1)

    # Fit the grid search to the training data
    grid_search.fit(X_train, y_train)

    # Predict on the test set using the best estimator
    best_pipeline = grid_search.best_estimator_

    return best_pipeline

def predict_student_class(pipeline, student_data, labels):
    prediction = pipeline.predict(student_data)
    return labels[prediction[0]]

def calculate_overall_percentile(data_point):
    return np.mean(data_point)

def identify_best_worst_attributes(data_point, attribute_names):
    best_index = np.argmax(data_point)
    worst_index = np.argmin(data_point)
    return attribute_names[best_index], attribute_names[worst_index], best_index, worst_index

def get_attribute_description(attribute, descriptions, best=True):
    return descriptions[attribute]['positive'] if best else descriptions[attribute]['negative']

def get_color(value):
    if value < 50:
        return f"rgb({255}, {int((value / 50) * 255)}, 0)"
    else:
        return f"rgb({int((1 - (value - 50) / 50) * 255)}, 255, 0)"

def create_percentile_gauge(title, value, domain):
    gradient_steps = []
    for i in range(100):
        if i < 50:
            color = f"rgba({255}, {int((i / 50) * 255)}, 0, 0.3)"  # Red to Yellow gradient with opacity
        else:
            color = f"rgba({int((1 - (i - 50) / 50) * 255)}, 255, 0, 0.3)"  # Yellow to Green gradient with opacity
        gradient_steps.append({'range': [i, i + 1], 'color': color})

    return go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': get_color(value), 'thickness': 0.3},  # Adjust bar thickness
            'bgcolor': 'white',  # White background for clarity
            'steps': gradient_steps,
        },
        domain=domain
    )

def plot_student_performance(overall_percentile, best_attribute, best_percentile, worst_attribute, worst_percentile, best_desc, worst_desc, prediction_label):
    fig = go.Figure()

    # Overall Percentile Gauge
    fig.add_trace(create_percentile_gauge("Overall Percentile", overall_percentile, {'x': [0, 0.4], 'y': [0.5, 1]}))

    # Best Attribute Gauge
    fig.add_trace(create_percentile_gauge(f"Best Attribute: {best_attribute}", best_percentile, {'x': [0.6, 1], 'y': [0.5, 1]}))

    # Worst Attribute Gauge
    fig.add_trace(create_percentile_gauge(f"Worst Attribute: {worst_attribute}", worst_percentile, {'x': [0.3, 0.7], 'y': [0, 0.4]}))

    fig.add_annotation(
        x=1.03, y=0.6,
        text=best_desc,
        showarrow=False,
        font=dict(size=12),
        align="center"
    )

    fig.add_annotation(
        x=0.5, y=0.03,
        text=worst_desc,
        showarrow=False,
        font=dict(size=12),
        align="center"
    )

    fig.update_layout(
        title='Student Performance Summary',
        autosize=False,
        width=1000,
        height=800,
        margin=dict(l=50, r=50, b=50, t=50),
        paper_bgcolor='#5dbdd6',
        plot_bgcolor='#5dbdd6',
    )
    fig.add_annotation(
        x=0.001, y=1.01,
        text=f"<b>Dyslexia Diagnosis:</b> {prediction_label}",
        showarrow=False,
        font=dict(size=14, color="black"),
        align="center",
        xref="paper",
        yref="paper",
        bordercolor="black",
        borderwidth=1,
        borderpad=10,
        bgcolor="lightyellow",
        opacity=0.8
    )
    
    # Render the plot to HTML
    return pio.to_html(fig, full_html=False)

# Initialize and train the model once when the application starts
def initialize_model(file_path):
    X, y = load_data(file_path)
    best_pipeline = train_model(X, y)
    return best_pipeline

# Load and initialize the trained model
trained_pipeline = initialize_model(file_path)

@app.route('/', methods=['GET', 'POST'])
def home():
    scores = {}
    errors = {}
    if request.method == 'POST':
        for category in categories:
            try:
                value = float(request.form.get(category, 0))
                if 0 <= value <= 100:
                    scores[category] = value
                else:
                    errors[category] = "Value must be between 0 and 100."
            except ValueError:
                errors[category] = "Please enter a valid number."

        if not errors:  # Proceed only if there are no errors
            student_data = np.array([list(scores.values())])

            # Make predictions
            prediction_label = predict_student_class(trained_pipeline, student_data, dyslexia_labels)
            overall_percentile = calculate_overall_percentile(student_data[0])
            
            # Identify best and worst attributes
            best_attribute, worst_attribute, best_index, worst_index = identify_best_worst_attributes(student_data[0], categories)
            best_attribute_percentile = student_data[0, best_index]
            worst_attribute_percentile = student_data[0, worst_index]

            # Get attribute descriptions
            best_attribute_description = get_attribute_description(best_attribute, attribute_descriptions, best=True)
            worst_attribute_description = get_attribute_description(worst_attribute, attribute_descriptions, best=False)

            # Generate the gauge HTML
            gauge_html = plot_student_performance(
                overall_percentile, best_attribute, best_attribute_percentile, 
                worst_attribute, worst_attribute_percentile, 
                best_attribute_description, worst_attribute_description, prediction_label
            )

            # Redirect to a new page displaying the results
            return render_template('index.html', scores=scores, prediction_label=prediction_label,
                                          overall_percentile=overall_percentile, best_attribute=best_attribute,
                                          worst_attribute=worst_attribute, best_attribute_percentile=best_attribute_percentile,
                                          worst_attribute_percentile=worst_attribute_percentile, best_desc=best_attribute_description,
                                          worst_desc=worst_attribute_description, gauge_html=gauge_html)

    return render_template('index.html', categories=categories, scores=scores, errors=errors)

if __name__ == '__main__':
    app.run(debug=True)