from flask import Flask, request, render_template
from services.api import API_Stock as api


app = Flask(__name__)


@app.route('/')
def hello_world():
    # return
    return render_template('index.html')


@app.route('/handle_data', methods=['GET'])
def handle_data():
    selected = request.args.get('stockList')
    symbolStock = selected + '.JKT'
    api.getByStockSymbolDaily(symbolStock)
#     return halaman yang menampilkan data grafik, tabel, result.html
    return

@app.route('/about')
def aboutPage():
    return render_template('about.html')

if __name__ == '__main__':
    app.run()
