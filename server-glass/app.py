from flask_apscheduler import APScheduler
from pathlib import Path
from kieta_modules.pipeline import PipelineManager
from posixpath import basename
from typing import Any, Dict, List
import uuid

from flask import Flask, render_template, request, Response, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging
import os
import ssl
import jsonpickle

from kieta_data_objs import Document

import sys
sys.path.append('kieta-modules')


logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(line1)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)

logger = logging.getLogger('main')


def pickle(obj: Any) -> str:
    return jsonpickle.encode(obj, make_refs=False, unpicklable=False)


def init_flask():
    app = Flask(__name__)
    CORS(app)
    logging.getLogger('flask_cors').level = logging.DEBUG

    # set config
    app.config.from_object(os.getenv("APP_SETTINGS"))

    if not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context
    return app


PATH = Path('/usr/src/server/')
TEMP_PATH = Path('/tmp/')

# Flask
app = init_flask()

pipelineManager: PipelineManager = PipelineManager()
pipelineManager.read_from_file(PATH / 'pipeline.json')


scheduler = APScheduler()
scheduler.api_enabled = True
scheduler.init_app(app)
scheduler.start()

jobs = {}
result = {}

####
# Flask Routes
####


def processDocumentForHTTPRequests_async(task_id, file_doc, pipeline):
    """
    Process a single document asynchronously
    File has to be saved in this case and cannot be streamed

    """
    jobs[task_id] = 'processing'

    # check if already in database and send overwrite notice
    res: Document = None
    doc: Dict[str, Any] = None

    with open(file_doc, 'rb') as file:
        # doc = {'file': file_doc.read(), 'id': ''.join(file_doc.filename.split('.')[:-1]), 'suffix':file_doc.filename.split('.')[-1]}
        doc = {'file': file.read(), 
               'id': ''.join(basename(file_doc).split('.')[:-1]),
               'suffix': basename(file_doc).split('.')[-1]
        }
    # delete file
    os.remove(file_doc)

    jobs[task_id] = 'processing'

    logger.info(f"Processing document {doc['id']}.{doc['suffix']}")
    try:
        res = pipelineManager.get_pipeline('NoOCR').process_full(doc)
        if pipeline == "OCR":
            res = pipelineManager.get_pipeline('OCR').process_full(res)
        res = pipelineManager.get_pipeline('GlassDigital').process_full(res)
    except Exception as e:
        logger.error(e)
        jobs[task_id] = 'failed'
        result[task_id] = e
        return e

    jobs[task_id] = 'done'
    result[task_id] = res

    return res


def processDocumentForHTTPRequests(doc, pipeline):
    """
    Process a single document
    """
    # check if already in database and send overwrite notice
    res: Document = None
    doc = {'file': doc.read(), 'id': ''.join(doc.filename.split('.')[:-1]), 'suffix':doc.filename.split('.')[-1]}
    logger.info(f"Processing document {doc['id']}.{doc['suffix']}")
    res = pipelineManager.get_pipeline('NoOCR').process_full(doc)
    if pipeline == "OCR":
        res = pipelineManager.get_pipeline('OCR').process_full(res)
    res = pipelineManager.get_pipeline('GlassDigital').process_full(res)
    return res


def handle_file(file, pipeline):
    task_id = str(uuid.uuid4())
    jobs[task_id] = 'queued'
    filename = secure_filename(file.filename)
    file_path = TEMP_PATH / filename
    file.save(file_path)

    scheduler.add_job(id=task_id, func=processDocumentForHTTPRequests_async, args=[
                      task_id, file_path, pipeline])
    return task_id


@app.route('/single_pdf_async', methods=['POST'])
def upload_file_async():
    try:
        if request.files['file']:
            task_id = handle_file(request.files['file'], "NoOCR")
            # j = processDocumentForHTTPRequests(request.files['file'], "NoOCR", None)
            # return Response(pickle(j), status=200, mimetype='application/json')
            return Response(pickle({'task_id': task_id}), status=200, mimetype='application/json')
        else:
            return Response(pickle({'status': 'error', 'msg': 'No file found!'}), status=400, mimetype='application/json')
    except Exception as e:
        logger.error(e)
        return Response(pickle({'status': 'error!', 'msg': e}), status=500, mimetype='application/json')


@app.route('/single_pdf_ocr_async', methods=['POST'])
def upload_file_ocr_async():
    try:
        if request.files['file']:
            task_id = handle_file(request.files['file'], "OCR")
            return Response(pickle({'task_id': task_id}), status=200, mimetype='application/json')
        else:
            return Response(pickle({'status':'error', 'msg': 'No file found!'}), status=400, mimetype='application/json')
    except Exception as e:
        logger.error(e)
        return Response(pickle({'status':'error','msg': e}), status=500, mimetype='application/json')


@app.route('/status/<task_id>', methods=['GET'])
def taskstatus(task_id):
    status = jobs.get(task_id, 'unknown')
    if status == 'done':
        del jobs[task_id]
        ret_obj = {'status': status, 'result': result[task_id]}
        del result[task_id]
        return Response(pickle(ret_obj), status=200, mimetype='application/json')
    elif status == 'failed':
        del jobs[task_id]
        ret_obj = {'status': status, 'result': result[task_id]}
        del result[task_id]
        return Response(pickle(ret_obj), status=500, mimetype='application/json')
    return jsonify({'status': status})


@app.route('/single_pdf', methods=['POST'])
def upload_file():
    try:
        if request.files['file']:
            j = processDocumentForHTTPRequests(request.files['file'], "NoOCR")
            return Response(pickle(j), status=200, mimetype='application/json')
        else:
            return Response({'status': 'No file found!'}, status=400, mimetype='application/json')
    except Exception as e:
        logger.error(e)
        return Response({'status': 'Error processing document!'}, status=500, mimetype='application/json')


@app.route('/single_pdf_ocr', methods=['POST'])
def upload_file_ocr():
    try:
        if request.files['file']:
            j = processDocumentForHTTPRequests(request.files['file'], "OCR")
            return Response(pickle(j), status=200, mimetype='application/json')
        else:
            return Response({'status': 'No file found!'}, status=400, mimetype='application/json')
    except Exception as e:
        logger.error(e)
        return Response({'status': 'Error processing document!'}, status=500, mimetype='application/json')


@app.route('/update_model', methods=['POST'])
def update_model():
    # Get the new model.pth from the request body
    try:
        new_model = request.files['file']
    except Exception as e:
        logger.error(e)
        return Response({'status': 'Error updating model!'}, status=500, mimetype='application/json')

    new_model.save(PATH / 'model.pth')

    # Update the model
    try:
        pipelineManager.clear()
        pipelineManager.read_from_file(PATH / 'pipeline.json')
    except Exception as e:
        logger.error(e)
        return Response({'status': 'Error updating model!'}, status=500, mimetype='application/json')
    logger.info("Model updated successfully!")
    return Response({'status': 'Model updated successfully!'}, status=200, mimetype='application/json')


@app.route('/update_pipeline', methods=['POST'])
def update_pipeline():
    # Get the new pipeline from the request body
    try:
        new_pipeline = request.files['file']
    except Exception as e:
        logger.error(e)
        return Response({'status': 'Error updating pipeline!'}, status=500, mimetype='application/json')

    new_pipeline.save(PATH / 'pipeline.json')

    # Update the pipeline
    try:
        pipelineManager.clear()
        pipelineManager.read_from_file(PATH / 'pipeline.json')
    except Exception as e:
        logger.error(e)
        return Response({'status': 'Error updating pipeline!'}, status=500, mimetype='application/json')

    logger.info("Pipeline updated successfully!")
    return Response({'status': 'Pipeline updated successfully!'}, status=200, mimetype='application/json')

@app.route('/show', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/update_knowledge', methods=['POST'])
def replace_knowledge():
    import json
    
    try:
        try:
            knowledge = request.json
        except Exception as e:
            # try if it's a file
            try:
                knowledge = request.files['file']
                knowledge = json.load(knowledge)
            except Exception as e:
                logger.error(e)
                return Response({'status': 'Malformed request!'}, status=400, mimetype='application/json')
        # save to json
        with open(PATH / 'knowledge.json', 'w') as f:
            json.dump(knowledge, f)
        
        pipelineManager.clear()
        pipelineManager.read_from_file(PATH / 'pipeline.json')

        return Response({'status': 'Knowledge updated successfully!'}, status=200, mimetype='application/json')
    except Exception as e:
        logger.error(e)
        return Response({'status': 'Error updating knowledge!'}, status=500, mimetype='application/json')

if __name__ == '__main__':
    app.run()
