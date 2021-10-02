import os
import sys
import uuid
import json
import inference
import datetime
from config import app
from models import db, Image
#from models import db, Image, Grade
from werkzeug.utils import secure_filename
from flask import render_template, flash, request, redirect, url_for, Flask, send_from_directory


db.create_all()


def allowed_files(filename):
    filename, extension = os.path.splitext(filename)
    return extension in app.config['ALLOWED_EXTENSIONS']

def create_day_folder(uploads):
    date = datetime.datetime.now()
    folder = f'{date.year}-{date.month}-{date.day}'
    if folder not in os.listdir(uploads):
        os.makedirs(os.path.join(uploads, folder))
    return folder

def dropdown():
    return

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    locations = ["1A","2A","3A", "4A", "5A", "6A", "7A", "8A","9A","10A","11A",
                    "12A","1B","2B","3B","4B","5B","6B", "7B", "8B", "9B","10B",
                 "11B", "12B", "13B", "14B","15B", "1C", "2C", "1G", "2G", "3G", "4G"]
    if request.method == 'POST':
        location = request.form.get('location')

        if 'file' not in request.files:
            flash('No file part.')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_files(file.filename):
            print(file.filename, file=sys.stderr)
            filename, file_extension = os.path.splitext(file.filename)
            file.filename = str(uuid.uuid4()) + file_extension
            filename = secure_filename(file.filename)
            directory = create_day_folder(app.config['UPLOAD_FOLDER'])
            filepath = os.path.join(directory, filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filepath))
            # call function that runs the neural network
            label = inference.predict_image(os.path.join(app.config['UPLOAD_FOLDER'], filepath), location)

            # redirect the user to the result of the network
            db.session.add(Image(filepath=filepath, location=location, label=label))
            #db.session.add(Image(filepath=filepath, location=location, label=label, name="Steve", grade=Grade.SENIOR))

            db.session.commit()
            return result(label)
        else:
            return file.filename
    else:
        return render_template("index.html", locations=locations)

@app.route('/data', methods=['GET'])
def get_data():
    return render_template("data.html", data=Image.query.all())

@app.route('/uploads/<path:filename>')
def download_file(filename):
    uploads = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'])
    return send_from_directory(uploads,
        filename, as_attachment=True)


@app.route('/', methods=['GET'])
def result(classified):
    return render_template("result.html", result=classified)


if __name__ == '__main__':
    app.run()
