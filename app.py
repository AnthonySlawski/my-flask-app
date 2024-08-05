from flask import Flask, render_template_string, request
import threading
from IPython.display import display, HTML
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

# Initialize file path for data
file_path = r"C:\Users\antho\OneDrive - University College Dublin\ACM20030 1\ALPACA\MASTER_DATA.csv"

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
    """
    Predicts the class for a new student using the provided pipeline.

    Parameters:
    pipeline: The trained model pipeline for predictions.
    student_data (np.array): The student data for prediction.
    labels (dict): Mapping of class indices to descriptive labels.

    Returns:
    str: The predicted class label.
    """
    prediction = pipeline.predict(student_data)
    return labels[prediction[0]]

def calculate_overall_percentile(data_point):
    """
    Calculates the overall percentile (mean of all attributes).

    Parameters:
    data_point (np.array): Array of student attributes.

    Returns:
    float: The mean percentile value.
    """
    return np.mean(data_point)

def identify_best_worst_attributes(data_point, attribute_names):
    """
    Identifies the best and worst attributes based on their values.

    Parameters:
    data_point (np.array): Array of student attributes.
    attribute_names (list): List of attribute names.

    Returns:
    tuple: Best attribute name, worst attribute name, best attribute index, worst attribute index.
    """
    best_index = np.argmax(data_point)
    worst_index = np.argmin(data_point)
    return attribute_names[best_index], attribute_names[worst_index], best_index, worst_index

def get_attribute_description(attribute, descriptions, best=True):
    """
    Retrieves the description of an attribute based on its performance.

    Parameters:
    attribute (str): The attribute name.
    descriptions (dict): Dictionary of attribute descriptions.
    best (bool): Whether to get the positive (best) or negative (worst) description.

    Returns:
    str: The description for the attribute.
    """
    return descriptions[attribute]['positive'] if best else descriptions[attribute]['negative']

def get_color(value):
    """
    Determines the color based on the percentile value.

    Parameters:
    value (float): The percentile value.

    Returns:
    str: The RGB color string.
    """
    if value < 50:
        return f"rgb({255}, {int((value / 50) * 255)}, 0)"
    else:
        return f"rgb({int((1 - (value - 50) / 50) * 255)}, 255, 0)"

def create_percentile_gauge(title, value, domain):
    """
    Creates a gauge for displaying a percentile value with a semi-transparent gradient background.

    Parameters:
    title (str): The title of the gauge.
    value (float): The value to display on the gauge.
    domain (dict): The domain for positioning the gauge.

    Returns:
    go.Indicator: A Plotly Indicator for the gauge.
    """
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
    """
    Plots the student's performance summary using gauges.

    Parameters:
    overall_percentile (float): Overall percentile score.
    best_attribute (str): Name of the best attribute.
    best_percentile (float): Percentile score of the best attribute.
    worst_attribute (str): Name of the worst attribute.
    worst_percentile (float): Percentile score of the worst attribute.
    best_desc (str): Description of the best attribute.
    worst_desc (str): Description of the worst attribute.
    prediction_label (str): Predicted class label.
    """
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
    # Load and preprocess data
    X, y = load_data(file_path)
    # Train the model
    best_pipeline = train_model(X, y)
    return best_pipeline

# Load and initialize the trained model
trained_pipeline = initialize_model(file_path)

# Define the home route
@app.route('/', methods=['GET', 'POST'])
def home():
    scores = {}
    errors = {}  # Dictionary to hold validation errors
    if request.method == 'POST':
        # Capture scores from the form
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
            return render_template_string(new_page_template, scores=scores, prediction_label=prediction_label,
                                          overall_percentile=overall_percentile, best_attribute=best_attribute,
                                          worst_attribute=worst_attribute, best_attribute_percentile=best_attribute_percentile,
                                          worst_attribute_percentile=worst_attribute_percentile, best_desc=best_attribute_description,
                                          worst_desc=worst_attribute_description, gauge_html=gauge_html)

    return render_template_string(home_page_template, categories=categories, scores=scores, errors=errors)

# New page template to display results
new_page_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Results - Alpaca Assessment Clone</title>
    <link rel="stylesheet" type="text/css" href="{{ url_for('static', filename='style.css') }}">
    <link href="https://fonts.googleapis.com/css?family=Montserrat:400,700|Open+Sans:400,600&display=swap" rel="stylesheet">
</head>
<body>
    <nav>
        <div class="nav-container">
            <a href="https://www.alpaca-assessment.com/" class="logo">
                <img src="https://www.alpaca-assessment.com/hs-fs/hubfs/alpaca_website_logo_homepage-1.png?width=3136&height=4258&name=alpaca_website_logo_homepage-1.png" alt="Alpaca Logo" class="logo-image" style="height: 50px;">
            </a>
            <span class="alpaca-text">ALPACA</span>
            <ul class="nav-links">
                <li><a href="https://www.alpaca-assessment.com/" target="_blank">Home</a></li>
                <li><a href="https://www.alpaca-assessment.com/about-us" target="_blank">About Us</a></li>
                <li><a href="https://26983596.hs-sites-eu1.com/faqs?__hstc=115865305.51ecb17916fe782745683a9d3ce90c82.1718311142824.1719928319519.1722861086113.6&__hssc=115865305.5.1722861086113&__hsfp=4143607224" target="_blank">FAQ</a></li>
            </ul>
        </div>
    </nav>
    <div class="content">
        <h1>Calculation Results</h1>
        <p><strong>Dyslexia Diagnosis:</strong> {{ prediction_label }}</p>
        <p><strong>Overall Percentile:</strong> {{ overall_percentile }}</p>
        <h2>Best and Worst Attributes</h2>
        <p><strong>Best Attribute:</strong> {{ best_attribute }} ({{ best_attribute_percentile }}%) - {{ best_desc }}</p>
        <p><strong>Worst Attribute:</strong> {{ worst_attribute }} ({{ worst_attribute_percentile }}%) - {{ worst_desc }}</p>
        
        <!-- Centering the Plotly plot -->
        <div class="plotly-plot-container">
            {{ gauge_html|safe }}
        </div>

        <!-- Interventions Button -->
        <div style="text-align: center; margin-top: 20px;">
            <a class="btn" href="https://26983596.hs-sites-eu1.com/join-us?__hstc=115865305.51ecb17916fe782745683a9d3ce90c82.1718311142824.1722868104107.1722876378018.8&__hssc=115865305.1.1722876378018&__hsfp=4143607224" target="_blank">Interventions</a>
        </div>
    </div>
</body>
</html>
'''

# Home page template
home_page_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Alpaca Assessment Clone</title>
    <link rel="stylesheet" type="text/css" href="{{ url_for('static', filename='style.css') }}">
    <link href="https://fonts.googleapis.com/css?family=Montserrat:400,700|Open+Sans:400,600&display=swap" rel="stylesheet">
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Get all input fields within the form
            const inputs = document.querySelectorAll('.diagnosis-tool input');

            // Add keydown event listener to each input
            inputs.forEach((input, index) => {
                input.addEventListener('keydown', function(event) {
                    if (event.key === 'Enter') {
                        event.preventDefault(); // Prevent form submission
                        // Move focus to the next input field if available
                        if (index < inputs.length - 1) {
                            inputs[index + 1].focus();
                        } else {
                            // Submit the form if it's the last input
                            input.form.submit();
                        }
                    }
                });
            });
        });
    </script>
</head>
<body>
    <nav>
        <div class="nav-container">
            <a href="https://www.alpaca-assessment.com/" class="logo">
                <img src="https://www.alpaca-assessment.com/hs-fs/hubfs/alpaca_website_logo_homepage-1.png?width=3136&height=4258&name=alpaca_website_logo_homepage-1.png" alt="Alpaca Logo" class="logo-image" style="height: 50px;">
            </a>
            <span class="alpaca-text">ALPACA</span>
            <ul class="nav-links">
                <li><a href="https://www.alpaca-assessment.com/" target="_blank">Home</a></li>
                <li><a href="https://www.alpaca-assessment.com/about-us" target="_blank">About Us</a></li>
                <li><a href="https://26983596.hs-sites-eu1.com/faqs?__hstc=115865305.51ecb17916fe782745683a9d3ce90c82.1718311142824.1719928319519.1722861086113.6&__hssc=115865305.5.1722861086113&__hsfp=4143607224" target="_blank">FAQ</a></li>
            </ul>
        </div>
    </nav>
    <div class="content">
        <h1>Welcome to Alpaca Assessment!</h1>
        <p>Explore our services and offerings with ease and elegance.</p>
        <a class="btn" href="https://www.alpaca-assessment.com/" target="_blank">Visit Alpaca Assessment</a>
        
        <h2>Dyslexia Diagnosis Tool</h2>
        <p>Enter your child's scores:</p>
        <form method="POST">
            <div class="diagnosis-tool">
                {% for category in categories %}
                <div class="score-box">
                    <label for="{{ category }}">{{ category }}</label>
                    <input type="number" id="{{ category }}" name="{{ category }}" value="{{ scores.get(category, '') }}" min="0" max="100" class="{% if errors.get(category) %}error{% endif %}">
                    {% if errors.get(category) %}
                    <div class="error-message">{{ errors[category] }}</div>
                    {% endif %}
                </div>
                {% endfor %}
            </div>
            <button type="submit" class="btn">Submit and Calculate</button>
        </form>
    </div>
</body>
</html>
'''

# Define a static route for CSS
@app.route('/static/style.css')
def stylesheet():
    return '''
    /* General styles */
    body {
        font-family: 'Open Sans', sans-serif;
        margin: 0;
        padding: 0;
        background-color: #5dbdd6;
        color: #ffffff;
    }

    /* Navigation styles */
    nav {
        background-color: #f8f9fa;
        border-bottom: 1px solid #eaeaea;
        position: fixed;
        width: 100%;
        top: 0;
        left: 0;
        z-index: 1000;
    }
    .nav-container {
        max-width: 1200px;
        margin: 0 auto;
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 20px;
    }
    .logo {
        display: flex;
        align-items: center;
        text-decoration: none;
    }
    .logo-image {
        height: 40px;
        margin-right: 10px;
    }
    .nav-links {
        list-style-type: none;
        padding: 0;
        margin: 0;
        display: flex;
        gap: 15px;
    }
    .nav-links li {
        display: inline;
    }
    .nav-links a {
        text-decoration: none;
        color: #333333;
        font-size: 1em;
        transition: color 0.3s;
    }
    .nav-links a:hover {
        color: #007bff;
    }

    /* Content styles */
    .content {
        max-width: 1200px;
        margin: 100px auto;
        padding: 20px;
        text-align: center;
    }
    h1, h2 {
        font-size: 2.5em;
        color: #ffffff;
        font-family: 'Montserrat', sans-serif;
    }
    h1 {
        font-family: 'Comic Sans MS', cursive, sans-serif; /* Change font to Comic Sans for the main title */
    }

    h2 {
        font-family: 'Comic Sans MS', cursive, sans-serif; /* Change font to Comic Sans for the tool title */
    }
    p {
        font-size: 1.2em;
        color: #555555;
    }
    .btn {
        display: inline-block;
        padding: 10px 30px;
        margin-top: 20px;
        background-color: #007bff;
        color: white;
        text-decoration: none;
        border-radius: 5px;
        transition: background-color 0.3s;
    }
    .btn:hover {
        background-color: #0056b3;
    }

    /* Diagnosis Tool Styles */
    .diagnosis-tool {
        display: flex; /* Use flexbox to align items in a single line */
        flex-wrap: wrap; /* Allow wrapping to the next line if needed */
        gap: 20px;
        margin-top: 20px;
        justify-content: space-between; /* Ensure spacing between items */
    }
    .score-box {
        display: flex;
        flex-direction: column;
        align-items: center;
        width: 150px; /* Set a fixed width for each score box */
        height: 120px; /* Set a fixed height to ensure uniformity */
        justify-content: space-between; /* Space out label and input vertically */
        text-align: center;
    }
    .score-box label {
        margin-bottom: 5px;
        font-weight: bold;
        font-size: 0.9em;
        height: 40px; /* Ensure labels occupy the same vertical space */
        display: flex;
        align-items: center; /* Center label text vertically */
        justify-content: center;
    }
    .score-box input {
        width: 100%;
        padding: 5px;
        font-size: 1em;
        box-sizing: border-box;
    }
    .score-box input.error {
        border: 2px solid red; /* Highlight errors with a red border */
    }
    .error-message {
        color: red;
        font-size: 0.8em;
        margin-top: 5px;
    }
    /* Text specific color changes */
    p.special-text {
        color: #d2e8c4; /* Set color for specific paragraphs */
    }

    /* Button color changes */
    .btn {
        background-color: #d2e8c4; /* Change button background color */
        color: #333333; /* Set text color for readability on buttons */
    }

    /* Style input boxes within the diagnosis tool */
    .diagnosis-tool input {
        background-color: #d2e8c4; /* Change background color for input boxes */
        border: none; /* Remove border to match style */
        color: #333333; /* Ensure input text is readable */
    }

    .diagnosis-tool input::placeholder {
        color: #333333; /* Ensure placeholder text is also visible */
    }
    /* Styling for the ALPACA text in the header */
    .alpaca-text {
        font-family: 'Comic Sans MS', cursive, sans-serif; /* Comic Sans font */
        color: #5dbdd6; /* Text color */
        font-size: 2em; /* Font size for emphasis */
        margin-left: 10px; /* Space between the logo and the text */
        vertical-align: middle; /* Align text with the logo */
    }    
    /* Style input boxes within the diagnosis tool */
    .diagnosis-tool input {
        background-color: #ffffff; /* Change background color to white */
        border: 1px solid #ccc; /* Add a light border for clarity */
        color: #333333; /* Ensure input text is readable */
        padding: 5px;
        border-radius: 4px; /* Add a slight border radius for style */
    }

    .diagnosis-tool input::placeholder {
        color: #666666; /* Placeholder text should be visible */
    }
    /* Center the Plotly plot container */
    .plotly-plot-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 20px auto;
        max-width: 1000px; /* Optional: Set a maximum width to ensure responsiveness */
    } 
    ''', 200, {'Content-Type': 'text/css'}

# Function to run the Flask app
def run_app():
    app.run(debug=True, use_reloader=False)

# Display the link to open the app in a new tab
display(HTML('<a href="http://127.0.0.1:5000" target="_blank">Open Flask App</a>'))

# Run the Flask app in a separate thread
thread = threading.Thread(target=run_app)
thread.start()