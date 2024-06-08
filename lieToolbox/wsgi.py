import os

from flask import Flask


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    # a simple page that says hello
    @app.route('/hello')
    def hello():
        return 'Hello, World!'

    from . import lie
    app.register_blueprint(lie.bp)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0")

