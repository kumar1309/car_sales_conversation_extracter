from flask import Flask, request, render_template, send_file, jsonify
import os
from pdfminer.high_level import extract_text
import re
from transformers import pipeline
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Preprocessing Functions

def extract_text_from_pdf(pdf_path):
    return extract_text(pdf_path)

def preprocess_text(text):
    # Remove special characters and extra spaces
    text = re.sub(r'\s+', ' ', text)
    return text

def load_transcript(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.txt'):
        with open(file_path, 'r') as file:
            return file.read()
    else:
        raise ValueError("Unsupported file format")

# Zero-Shot Learning for Information Extraction

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def extract_information(text, labels):
    result = classifier(text, labels)
    return {label: result['scores'][result['labels'].index(label)] if label in result['labels'] else None for label in labels}

def extract_customer_requirements(text):
    labels = ["Hatchback", "SUV", "Sedan", "Petrol", "Diesel", "Electric", "Red", "Blue", "White", "Black",
              "Manual", "Automatic"]
    return extract_information(text, labels)

def extract_company_policies(text):
    labels = ["Free RC Transfer", "5-Day Money Back Guarantee", "Free RSA for One Year", "Return Policy"]
    return extract_information(text, labels)

def extract_customer_objections(text):
    labels = ["Refurbishment Quality", "Car Issues", "Price Issues", "Customer Experience Issues"]
    return extract_information(text, labels)

# Structuring Output in JSON Format

def structure_output(requirements, policies, objections, car_colors, price_ranges, car_types, refurbishment_issues, frequent_objections):
    output = {
        "Customer Requirements": requirements,
        "Company Policies Discussed": policies,
        "Customer Objections": objections,
        "Car Colors": car_colors,
        "Price Ranges": price_ranges,
        "Car Types": car_types,
        "Refurbishment Issues": refurbishment_issues,
        "Frequent Objections": frequent_objections
    }
    return output

def process_transcript(file_path):
    transcript = preprocess_text(load_transcript(file_path))
    requirements = extract_customer_requirements(transcript)
    policies = extract_company_policies(transcript)
    objections = extract_customer_objections(transcript)
    # Dummy data for new categories (update these functions with actual extraction logic if available)
    car_colors = {"Red": 10, "Blue": 5, "White": 8, "Black": 7}
    price_ranges = {"Below $20,000": 12, "$20,000 - $30,000": 15, "Above $30,000": 8}
    car_types = {"SUV": 20, "Sedan": 10, "Hatchback": 7}
    refurbishment_issues = {"Paint Quality": 5, "Engine Issues": 3, "Interior Issues": 2}
    frequent_objections = {"Price": 8, "Quality": 5, "Features": 3}

    return structure_output(requirements, policies, objections, car_colors, price_ranges, car_types, refurbishment_issues, frequent_objections)

def vis(data):
    # Initialize empty dataframes for aggregation
    requirements_df = pd.DataFrame(columns=["Requirement", "Score"])
    objections_df = pd.DataFrame(columns=["Objection", "Score"])
    colors_df = pd.DataFrame(columns=["Color", "Count"])
    price_ranges_df = pd.DataFrame(columns=["Price Range", "Count"])
    car_types_df = pd.DataFrame(columns=["Type", "Count"])
    refurbishment_df = pd.DataFrame(columns=["Issue", "Count"])
    objections_freq_df = pd.DataFrame(columns=["Objection", "Count"])

    # Aggregate data from all transcripts
    for entry in data:
        req_df = pd.DataFrame(list(entry["Customer Requirements"].items()), columns=["Requirement", "Score"])
        obj_df = pd.DataFrame(list(entry["Customer Objections"].items()), columns=["Objection", "Score"])
        colors_df = pd.concat([colors_df, pd.DataFrame(list(entry["Car Colors"].items()), columns=["Color", "Count"])])
        price_ranges_df = pd.concat([price_ranges_df, pd.DataFrame(list(entry["Price Ranges"].items()), columns=["Price Range", "Count"])])
        car_types_df = pd.concat([car_types_df, pd.DataFrame(list(entry["Car Types"].items()), columns=["Type", "Count"])])
        refurbishment_df = pd.concat([refurbishment_df, pd.DataFrame(list(entry["Refurbishment Issues"].items()), columns=["Issue", "Count"])])
        objections_freq_df = pd.concat([objections_freq_df, pd.DataFrame(list(entry["Frequent Objections"].items()), columns=["Objection", "Count"])])

        # Concatenate individual dataframes into main dataframes
        requirements_df = pd.concat([requirements_df, req_df])
        objections_df = pd.concat([objections_df, obj_df])

    # Aggregate scores and counts
    requirements_df = requirements_df.groupby("Requirement", as_index=False).mean()
    objections_df = objections_df.groupby("Objection", as_index=False).mean()
    colors_df = colors_df.groupby("Color", as_index=False).sum()
    price_ranges_df = price_ranges_df.groupby("Price Range", as_index=False).sum()
    car_types_df = car_types_df.groupby("Type", as_index=False).sum()
    refurbishment_df = refurbishment_df.groupby("Issue", as_index=False).sum()
    objections_freq_df = objections_freq_df.groupby("Objection", as_index=False).sum()

    # Save the plots as images
    def save_plot(df, x_col, y_col, filename, title):
        plt.figure(figsize=(10, 6))
        sns.barplot(x=x_col, y=y_col, data=df, palette='Blues_d' if 'Score' in y_col else 'Reds_d')
        plt.title(title)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        file_path = os.path.join('static', filename)
        plt.savefig(file_path)
        plt.close()
        return file_path

    requirements_image_path = save_plot(requirements_df, 'Score', 'Requirement', 'requirements.png', 'Customer Requirements Scores')
    objections_image_path = save_plot(objections_df, 'Score', 'Objection', 'objections.png', 'Customer Objections Scores')
    car_colors_image_path = save_plot(colors_df, 'Count', 'Color', 'car_colors.png', 'Distribution of Car Colors')
    price_ranges_image_path = save_plot(price_ranges_df, 'Count', 'Price Range', 'price_ranges.png', 'Popular Price Ranges')
    car_types_image_path = save_plot(car_types_df, 'Count', 'Type', 'car_types.png', 'Preferred Car Types')
    refurbishment_issues_image_path = save_plot(refurbishment_df, 'Count', 'Issue', 'refurbishment_issues.png', 'Common Refurbishment Issues')
    frequent_objections_image_path = save_plot(objections_freq_df, 'Count', 'Objection', 'frequent_objections.png', 'Frequently Raised Objections')

    return (requirements_image_path, objections_image_path, car_colors_image_path, price_ranges_image_path,
            car_types_image_path, refurbishment_issues_image_path, frequent_objections_image_path)

# Export Functionality
def export_to_csv(data):
    requirements_df = pd.DataFrame([entry["Customer Requirements"] for entry in data])
    objections_df = pd.DataFrame([entry["Customer Objections"] for entry in data])

    requirements_csv = os.path.join('static', 'requirements.csv')
    objections_csv = os.path.join('static', 'objections.csv')

    requirements_df.to_csv(requirements_csv, index=False)
    objections_df.to_csv(objections_csv, index=False)

    return requirements_csv, objections_csv

def export_car_colors_csv(data):
    colors_df = pd.DataFrame([entry["Car Colors"] for entry in data])
    car_colors_csv = os.path.join('static', 'car_colors.csv')
    colors_df.to_csv(car_colors_csv, index=False)
    return car_colors_csv

def export_price_ranges_csv(data):
    price_ranges_df = pd.DataFrame([entry["Price Ranges"] for entry in data])
    price_ranges_csv = os.path.join('static', 'price_ranges.csv')
    price_ranges_df.to_csv(price_ranges_csv, index=False)
    return price_ranges_csv

def export_car_types_csv(data):
    car_types_df = pd.DataFrame([entry["Car Types"] for entry in data])
    car_types_csv = os.path.join('static', 'car_types.csv')
    car_types_df.to_csv(car_types_csv, index=False)
    return car_types_csv

def export_refurbishment_issues_csv(data):
    refurbishment_df = pd.DataFrame([entry["Refurbishment Issues"] for entry in data])
    refurbishment_issues_csv = os.path.join('static', 'refurbishment_issues.csv')
    refurbishment_df.to_csv(refurbishment_issues_csv, index=False)
    return refurbishment_issues_csv

def export_frequent_objections_csv(data):
    objections_freq_df = pd.DataFrame([entry["Frequent Objections"] for entry in data])
    frequent_objections_csv = os.path.join('static', 'frequent_objections.csv')
    objections_freq_df.to_csv(frequent_objections_csv, index=False)
    return frequent_objections_csv

def export_to_pdf(data):
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Car Sales Analysis Report", ln=True, align='C')
    pdf.ln(10)

    # Add text content
    pdf.multi_cell(0, 10, txt=json.dumps(data, indent=4))
    
    pdf_path = os.path.join('static', 'report.pdf')
    pdf.output(pdf_path)

    return pdf_path

# Flask App Setup

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        files = request.files.getlist('files')  # Get the list of files
        if files:
            all_data = []
            for file in files:
                # Save each uploaded file to the uploads directory
                file_path = os.path.join('uploads', file.filename)
                file.save(file_path)

                # Process each transcript and collect the result
                data = process_transcript(file_path)
                all_data.append(data)

            # Generate visualizations for all uploaded files
            (requirements_image_path, objections_image_path, car_colors_image_path, price_ranges_image_path,
             car_types_image_path, refurbishment_issues_image_path, frequent_objections_image_path) = vis(all_data)

            # Export analysis to CSV and PDF
            requirements_csv, objections_csv = export_to_csv(all_data)
            car_colors_csv = export_car_colors_csv(all_data)
            price_ranges_csv = export_price_ranges_csv(all_data)
            car_types_csv = export_car_types_csv(all_data)
            refurbishment_issues_csv = export_refurbishment_issues_csv(all_data)
            frequent_objections_csv = export_frequent_objections_csv(all_data)
            pdf_path = export_to_pdf(all_data)

            # Render the images and data on the web page
            return render_template('results.html', 
                                   requirements_image=requirements_image_path,
                                   objections_image=objections_image_path,
                                   car_colors_image=car_colors_image_path,
                                   price_ranges_image=price_ranges_image_path,
                                   car_types_image=car_types_image_path,
                                   refurbishment_issues_image=refurbishment_issues_image_path,
                                   frequent_objections_image=frequent_objections_image_path,
                                   data=json.dumps(all_data, indent=4),
                                   requirements_csv=requirements_csv,
                                   objections_csv=objections_csv,
                                   car_colors_csv=car_colors_csv,
                                   price_ranges_csv=price_ranges_csv,
                                   car_types_csv=car_types_csv,
                                   refurbishment_issues_csv=refurbishment_issues_csv,
                                   frequent_objections_csv=frequent_objections_csv,
                                   pdf_path=pdf_path)
        return "No files uploaded", 400

if __name__ == '__main__':
    app.run(debug=True)
