import json

from flask import Flask, Response, request
from flask_restful import Api, Resource
from flask_cors import CORS
from transformers import pipeline

from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop

from ml_section.model_run.run_model import run_model


app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
api = Api(app)


def summarize(text):
    summarizer = \
        pipeline("summarization",
                 model="./results/checkpoint-843",
                 tokenizer="t5-base")

    # o modelo t5 precisa de ter "summarize: " como prefixo p sumarizar
    t5_input = "summarize: " + text

    summary = summarizer(t5_input,
                         min_length=30,
                         max_length=150,
                         num_beams=4,
                         early_stopping=True)

    summarized_text = summary[0]["summary_text"]

    return summarized_text


class getModelPrediction(Resource):
    def post(self):

        request_data = request.json

        if not isinstance(request_data["Title"], str) or\
           not isinstance(request_data["Text"], str):
            response = {"success": False,
                        "message": "Missing required fields."}
            return Response(json.dumps(response), 404)

        try:
            prediction, label = run_model(request_data)
            message = {"prediction": prediction,
                       "label": label}
            print(f'The NEWS article provided is considered {label} with a certanty of {prediction}')  # noqa: E501

            response = {"success": True,
                        "message": message}

            return Response(json.dumps(response), 200)

        except Exception as e:
            return Response(json.dumps(e), 404)


class getTextShort(Resource):
    def post(self):

        request_data = request.json

        if not isinstance(request_data["Text"], str):
            response = {"success": False,
                        "message": "Missing required fields."}
            return Response(json.dumps(response), 404)

        try:
            summary = summarize(request_data["Text"])
            response = {"success": True,
                        "message": summary}

            return Response(json.dumps(response), 200)

        except Exception as e:
            return Response(json.dumps(e), 404)


class Home(Resource):
    def get(self):
        response = {'status': 'Running'}
        return Response(json.dumps(response), status=200)


api.add_resource(Home, "/")
api.add_resource(getModelPrediction, "/get_prediction")
api.add_resource(getTextShort, "/get_summary")


def launch_api():
    app.run(port=4405)
    # from tornado.log import enable_pretty_logging
    # enable_pretty_logging()

    # http_server = HTTPServer(WSGIContainer(app))
    # http_server.listen(4405)

    # IOLoop.instance().start()
