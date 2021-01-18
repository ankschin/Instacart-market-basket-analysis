from flask import Flask, jsonify, request
import numpy as np
import pickle as pkl
from model import *
import time


import flask
app= Flask(__name__)

@app.route('/index', methods=['POST','GET'])
def index():
	if request.method =='GET':
		return flask.render_template('index.html')
	else:
		pred_list= request.form.to_dict()
		print(pred_list['user_id'],flush=True)
		user_id= str(pred_list['user_id'])
	try:
		user_id = int(user_id)
		print(user_id)
		return predict(user_id)
	except ValueError:
		# Handle the exception
		message= 'Login id should be a number. Please re-enter.'
		print(message)
		return flask.render_template('index.html', message=message)



@app.route('/')
def hello_world():
	return 'hello world!'


def predict(user_id):
	curr_time= time.time()
	predicted_products= final_fun_1(int(user_id))
	time_taken= time.time()-curr_time
	print('time taken = {}'.format(time_taken))
	return flask.render_template("result.html", result = predicted_products, time=time_taken)
	#return jsonify({'pred':predicted_products})


@app.route('/calculate')
def calculate():
	with open('X_test.pkl', 'rb') as f:
		X_test=pkl.load(f)
	#X_test=X_test.drop('reordered', axis=1)
	with open('y_test.pkl', 'rb') as f:
		y_test= pkl.load(f)
	score= final_fun_2(X_test,y_test)
	return 'F1 score is {}'.format(score)



if __name__=='__main__':
	app.run(host='0.0.0.0', port=8080)