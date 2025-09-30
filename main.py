# main.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from src.llm_response import LLMInvoke
import uuid
from logger_setup import setup_logger, start_request_logging, end_request_logging
import time
import os

# Make sure logs directory exists
os.makedirs("logs", exist_ok=True)

# Logger setup
logger = setup_logger(__name__)

# Flask app setup
app = Flask(__name__)
CORS(app)

@app.route('/')
def index() -> str:
    """Serves the HTML UI."""
    logger.info("Serving index.html to user")
    return render_template('index.html')

@app.route('/response', methods=['POST'])
def llm_response():
    request_id = str(uuid.uuid4())[:8]
    log_file = start_request_logging(request_id)
    start_time = time.time()
    
    logger.info(f"[{request_id}] New query request received")
   
    try:
        if not request.is_json:
            logger.warning(f"[{request_id}] Invalid request: Not JSON")
            return jsonify({"error": "Request must be JSON"}), 400

        query = request.get_data().decode("utf-8")
        logger.info(f"[{request_id}] Processing query: \"{query[:50]}{'...' if len(query) > 50 else ''}\"")

        llm_start_time = time.time()
        llm_invoke = LLMInvoke()
        response = llm_invoke.llm_response(query)
        llm_duration = time.time() - llm_start_time

        logger.info(f"[{request_id}] LLM response received in {llm_duration:.2f} seconds")
        logger.info(f"[{request_id}] Answer generated with {len(response.get('answer', ''))} characters")
        logger.info(f"[{request_id}] Processed {len(response.get('sources', []))} source documents")
        logger.debug(f"[{request_id}] Final response: {response}")

        total_duration = time.time() - start_time
        logger.info(f"[{request_id}] Total request processed in {total_duration:.2f} seconds")
        

        return jsonify({
            'answer': response['answer'],
            'sources': response['sources']
        })

    except KeyError as ke:
        error_msg = f"Missing key in response: {str(ke)}"
        logger.error(f"[{request_id}] {error_msg}")
        return jsonify({"error": error_msg}), 500

    except Exception as e:
        logger.error(f"[{request_id}] Error processing request: {str(e)}", exc_info=True)
        return jsonify({"error": "An error occurred while processing the request."}), 500

    finally:
        # Always end request logging
        end_request_logging(request_id)
        logger.info(f"[{request_id}] Request handling completed")

# Entry Point
if __name__ == '__main__':
    logger.info("Starting Flask app on port 5008...")
    app.run(port=5008, debug=True, use_reloader=True)