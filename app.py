import os
import threading
import torch
import logging
from flask import Flask, request, jsonify
import sys
import traceback
#sys.path.append(os.path.abspath('./src'))

from embed import test_text_embed
from extract import test_text_extract
current_dir = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(current_dir, 'output', 'exec_log.log')
app = Flask(__name__)

# 配置日志文件和日志级别
logging.basicConfig(filename=log_path , level=logging.INFO, 
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

@app.route('/embed', methods=['POST'])
def text_embed_interface():
    data = request.json
    task_id = data.get('task_id', 'unknown')
    message_paths = data.get('message_paths', 'unknown')
    output = data.get('output', 'unknown') 
    logging.info(f"Embed task {task_id} exec, message path is {message_paths}, output path is {output}.")
    try:
        test_text_embed(data)
        logging.info(f"Embed task {task_id} completed successfully.")
        return jsonify({"status": f"Embed task {task_id} started successfully"}), 200
    except Exception as e:
        error_message = traceback.format_exc()  # 获取详细的异常信息
        print(error_message)
        logging.error(f"Embed task {task_id} failed. Error: {e}")
        return jsonify({"status": f"Embed task {task_id} failed"}), 500

@app.route('/extract', methods=['POST'])
def text_extract_interface():
    data = request.json
    task_id = data.get('task_id', 'unknown')
    message_paths = data.get('message_paths', 'unknown')
    output = data.get('output', 'unknown') 
    logging.info(f"Extract task {task_id} exec, message path is {message_paths}, output path is {output}.")
    try:
        test_text_extract(data)
        logging.info(f"Extract task {task_id} completed successfully.")
        return jsonify({"status": f"Extract task {task_id} started successfully"}), 200
    except Exception as e:
        logging.error(f"Extract task {task_id} failed. Error: {e}")
        return jsonify({"status": f"Extract task {task_id} failed"}), 500


@app.route('/param_num', methods=['GET'])
def get_param_num():
    # To-Do: 获取算法模型的总参数
    # 若此处没有正确的设置，“模型参数量”指标将没有得分
    param_num = 58445114
    return jsonify({"data": f"param_num: {param_num}"}), 200


if __name__ == '__main__':
    #for rule in app.url_map.iter_rules():
    #    print(rule)  # 打印所有已注册的路由
    # 端口号可以自行修改
    app.run(host='0.0.0.0', port=51608)
