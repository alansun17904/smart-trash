from flask_sqlalchemy import SQLAlchemy
from config import app
from datetime import datetime


db = SQLAlchemy(app)

class Image(db.Model):
    id = Column(db.Integer, primary_key=True)
    filename = Column(db.String(50), unique=True, nullable=False)
    date = Column(db.DateTime, default=datetime.utcnow)
    label = Column(db.String(50))

    def __repr__(self):
        return f'<Image {filename}>'
