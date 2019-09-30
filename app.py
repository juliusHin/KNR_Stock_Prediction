from flask import Flask, request, render_template, Markup
from plotly.offline import plot

from services.api import API_Stock

app = Flask(__name__, static_url_path='/static')


@app.route('/')
def hello_world():
    # return
    return render_template('index.html')


@app.route('/result', methods=['GET'])
def handle_data():
    selected = request.args.get('stockList')
    # k = request.args.get('kvalue',default=1, type=int)
    symbol = selected + '.JKT'
    api = API_Stock(symbol)
    data_graph, data_table = api.getResult()
    plot_div = plot(data_graph, output_type='div')
    return render_template('result.html', title=str(selected), data_graph=Markup(plot_div),
                           table=data_table.to_html(justify='left'))

@app.route('/about')
def aboutPage():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
