from flask import Flask, render_template, request, url_for, send_from_directory
import requests
import json
import numpy as np
import os

app = Flask(__name__)

# define app url
if 'MY_CONNECTION_SETTING__serviceUri' in os.environ :
  # Running in Azure
  main_URL = "https://ocp9app.azurewebsites.net"
else:
  # Running locally 
  main_URL = "http://localhost:7071"

def get_list_if_ids() :
  """get the list of user_id s using azure function"""
  # trigger function "listofids"
  response = requests.post(
    url = main_URL+"/listofids",
    headers={"accept" : "application/json"}
  )
  # extract the list
  listofids = response.json()["list"]

  return listofids

def get_recs(user_id, n_cf) : 
  """get recommendations using azure function"""
  # triger function "recsfrommodel"
  # wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
  print("track - flask - get_recs/user_id", user_id)
  print("track - flask - get_recs/n_cf", n_cf)
  response = requests.get(
    # json={"user_id" : user_id, "n_cf" : n_cf},
    url = main_URL+"/recsfrommodel/"+str(user_id)+"/"+str(n_cf),
    headers={"accept" : "application/json"}
  )
  print("track - flask - get_recs/reponse", response)
  # extract the list
  recs = response.json()["recs"]

  return recs

@app.route('/index')
def index():
  """Show a cell to enter a user_id"""
  listofids = get_list_if_ids()

  return render_template('index.html', listofids=listofids)

@app.route('/result', methods=["POST"])
def result():
  """Show the user's top 5 recommendations"""
  if request.method == 'POST':
    # get list of ids 
    listofids = get_list_if_ids()
    # get user_id
    user_id = int(request.form['user_id'])
    n_cf = int(request.form['n_cf'])
    # n_cf = 2
    # wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
    print("track - flask - result/user_id", user_id)
    print("track - flask - result/n_cf", n_cf)
    if user_id in listofids:
      # get recommendations
      recs = get_recs(user_id=user_id, n_cf=n_cf)
      # wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
      print("track - flask - result/recs", recs)
      return render_template('result.html', user_id=user_id, n_cf=n_cf, listofids=listofids, recs=recs)
    else:
      error="bad user_id"
      return render_template('index.html', listofids=listofids, error=error)
    

# @app.route('/index')
# def index():
#   """Show a cell to enter a user_id"""
#   # listofids = get_list_if_ids()
#   listofids = list(range(20))

#   return render_template('index.html', listofids=listofids)

# @app.route('/result', methods=["POST"])
# def result():
#   """Show the user's top 5 recommendations"""
#   if request.method == 'POST':
#     # get list of ids 
#     # listofids = get_list_if_ids()
#     listofids = list(range(20))
#     # get user_id
#     user_id = int(request.form['user_id'])
#     n_cf = int(request.form['n_cf'])
#     # n_cf = 2
#     # wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
#     print("track - result/user_id", user_id)
#     print("track - result/n_cf", n_cf)
#     if user_id in listofids:
#       # get recommendations
#       # recs = get_recs(user_id=user_id, n_cf=n_cf)
#       recs = [user_id, n_cf]
#       # wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
#       print("track - result/recs", recs)
#       return render_template('result.html', user_id=user_id, listofids=listofids, recs=recs)
#     else:
#       error="bad user_id"
#       return render_template('index.html', listofids=listofids, error=error)



if __name__ == '__main__':
  app.run(debug=True)
