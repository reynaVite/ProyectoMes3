from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Jesus Antonio Ramirez Hernandez \n 20210700 \n 9 B "

if __name__ =='__main__':
    app.run(debug=True)