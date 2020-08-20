import os
from flask import render_template, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from config import app
# from models import db, Image


# db.create_all()

def allowed_files(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part.')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_files(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # adding image into the database for later
            i = Image(filename=filename)
            db.session.add(i)
            # call function that runs the neural network
            # redirect the user to the result of the network
            return redirect(url_for('upload_image',
                                        filename=filename))
    else:
        return '''
                <!doctype html>
                <title>ST!</title>
                <h1>Upload Image</h1>
                <form method=post enctype=multipart/form-data>
                  <input type=file name=file>
                  <input type=submit value=Upload>
                </form>
                '''
