#!/usr/bin/env python3

import os
import cv2
import numpy as np
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "supersecretkey"


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def detect_ellipses(image_path):
    """Detect ellipses in the uploaded image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred_image = cv2.GaussianBlur(image, (11, 11), 1)
    edges = cv2.Canny(blurred_image, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ellipse_count = 0

    for contour in contours:
        if len(contour) >= 5:
            cv2.fitEllipse(contour)
            ellipse_count += 1

    return ellipse_count


def detect_circles(image_path):
    """Detect circles in the uploaded image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred_image = cv2.GaussianBlur(image, (9, 9), 2)

    circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=50, param2=30, minRadius=0, maxRadius=0)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        return len(circles)

    return 0


def detect_squares_and_rectangles(image_path):
    """Detect squares and rectangles in the uploaded image."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    shape_count = 0

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            shape_count += 1

    return shape_count


def detect_triangles(image_path):
    """Detect triangles in the uploaded image."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    shape_count = 0

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        if len(approx) == 3:
            shape_count += 1

    return shape_count


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """Handle the file upload and shape detection."""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        shape = request.form.get('shape')

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            if shape == 'ellipse':
                shape_count = detect_ellipses(file_path)
                flash(f'Ellipses detected: {shape_count}')
            elif shape == 'circle':
                shape_count = detect_circles(file_path)
                flash(f'Circles detected: {shape_count}')
            elif shape in ['square', 'rectangle']:
                shape_count = detect_squares_and_rectangles(file_path)
                flash(f'{shape.capitalize()}s detected: {shape_count}')
            elif shape == 'triangle':
                shape_count = detect_triangles(file_path)
                flash(f'Triangles detected: {shape_count}')

            return redirect(url_for('upload_file'))

        flash('File type not allowed')
        return redirect(request.url)

    return render_template('index.html')


@app.route('/index.html')
def render_index():
    """Render the index page."""
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
