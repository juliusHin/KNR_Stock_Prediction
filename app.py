from flask import Flask, request, render_template
from services.api import API_Stock
import os
import time
from DateTime import DateTime


app = Flask(__name__)


@app.route('/')
def hello_world():
    # return
    return render_template('index.html')


@app.route('/handle_data', methods=['GET'])
def handle_data():
    selected = request.args.get('stockList')
    data = selected + '.JKT'
    api = API_Stock()
    data_graph, data_table = api.knn_stock_prediction(symbolStock = data)
#     return halaman yang menampilkan data grafik, tabel, result.html
    return render_template('result.html', selected, data_graph, data_table)

@app.route('/about')
def aboutPage():
    return render_template('about.html')

if __name__ == '__main__':
    app.run()
