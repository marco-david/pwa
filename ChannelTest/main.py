from flask import Flask, render_template, request, Response, make_response, jsonify, abort, send_from_directory, redirect, url_for, send_file
# from werkzeug import secure_filename
import os
import csv
import codecs
import datetime as dt
import sys
import datetime
import re



app = Flask(__name__)

@app.route('/')
def login():
   return render_template('main.html')


@app.route('/<path:filename>')
def custom_static(filename):
    return send_from_directory("static/", filename)




if __name__ == '__main__':
    # 6246 == MAIN;
    app.run(host='127.0.0.1', port=8080, debug=True)



