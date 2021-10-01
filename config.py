from flask import Flask


app = Flask('app')
app.secret_key = 'super secret key!'  # change to file in production
app.config['SESSION_TYPE'] = 'filesystem'
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/test.db'
app.config['ALLOWED_EXTENSIONS'] = set(['.jpg', '.jpeg', '.png'])
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.debug = True

if __name__ == '__main__':
	app.run()
