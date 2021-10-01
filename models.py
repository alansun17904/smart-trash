from flask_sqlalchemy import SQLAlchemy
from config import app
from datetime import datetime


db = SQLAlchemy(app)

class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filepath = db.Column(db.String(50), unique=True, nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)
    location = db.Column(db.String(50))
    label = db.Column(db.String(50))

    def __repr__(self):
        return f'<Image {self.filepath}>'
