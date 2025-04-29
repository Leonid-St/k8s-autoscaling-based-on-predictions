import os
from flask import Flask
import datetime
from loguru import logger
import hashlib
from pathlib import Path
import glob
import atexit
from flask import Response
import sys, socket

app = Flask(__name__)


@app.route("/")
def run():
    time = str(datetime.datetime.utcnow())
    logger.debug("Serving a request at " + time)
    return_msg = "Current UTC time is {} at host {} ".format(time, os.getenv('HOSTNAME', "unknown"))
    return Response(return_msg, status=200, mimetype='text/plain')


@app.route("/health")
def health():
    return Response("Works", status=200, mimetype='text/plain')

def test_is_prime(num):
    if num > 1:
        # Iterate from 2 to n / 2
        for i in range(2, num // 2):

            # If num is divisible by any number between
            # 2 and n / 2, it is not prime
            if (num % i) == 0:
                return False
        else:
            return True
    else:
        return False


@app.route("/prime/<num>", methods=['GET','POST','DELETE','UPDATE'])
def is_prime(num):
    return Response("True" if test_is_prime(int(num)) else "False", status=200, mimetype='text/plain')


@app.route('/hash/<id>', methods=['GET','POST','DELETE','UPDATE'])
def handle_post(id: str):
    hash = hashlib.md5(id.encode()).hexdigest()
    return Response(hash, status=200, mimetype='text/plain')


if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=int(os.environ.get("PUBLISH_PORT", 9000)))
    except KeyboardInterrupt:
        # clean_up_files()
        raise

