from flask import Flask, request, render_template, Markup
from plotly.offline import plot

from services.api import API_Stock

app = Flask(__name__)


@app.route('/')
def hello_world():
    # return
    return render_template('index.html')


@app.route('/result', methods=['GET'])

def handle_data():
    selected = request.args.get('stockList')

    symbol = selected + '.JKT'
    api = API_Stock(symbol, 6)
    data_graph = api.getResult()
    plot_div = plot(data_graph, output_type='div')
    return render_template('result.html', title=str(selected), data_graph=Markup(plot_div))

@app.route('/about')
def aboutPage():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
