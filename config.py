from flask import Flask


app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/test.db'
app.config['ALLOWED_EXTENSIONS'] = set(['jpg', 'jpeg', 'png'])
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
