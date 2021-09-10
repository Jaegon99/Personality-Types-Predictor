import webbrowser
from threading import Timer
from flask import Flask, render_template, request, redirect, url_for

from twitter import getType

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        yourType = ""
        handle = request.form["handle"]
        print(">>> " + handle)
        if handle != "":
            try:
                yourType = getType(handle)
            except Exception as e:
                print(">>> " + yourType)
                yourType = "error"
            return render_template('index.htm', user=handle, type=yourType.lower())
    return render_template('index.htm', user='', type='none')


def open_browser():
    webbrowser.open_new('http://127.0.0.1:2000/')


if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run(port=2000, debug=False)
