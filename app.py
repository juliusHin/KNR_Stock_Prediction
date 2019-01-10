from flask import Flask, request, render_template


app = Flask(__name__)


@app.route('/')
def hello_world():
    # return
    return render_template('index.html')


@app.route('/handle_data', methods=['GET'])
def handle_data():
    selected = request.args.get('stockList')
    return (str(selected))

@app.route('/about')
def aboutPage():
    return render_template('about.html')

if __name__ == '__main__':
    app.run()
