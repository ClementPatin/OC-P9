import azure.functions as func
import json
from FlaskApp.flask_app import app as flask_app
import utils.my_functions_for_prod as mf
from joblib import load
from io import BytesIO
import numpy as np
import os

blob_path = "ocp9container/"

def get_connection_setting():
    """
    Determines the connection setting to use based on the environment.

    Returns:
        str: The name of the connection setting to use. 
             - "MY_CONNECTION_SETTING" if running in Azure.
             - "AzureWebJobsStorage" if running locally.
    """
    if 'MY_CONNECTION_SETTING__serviceUri' in os.environ :
        # Running in Azure
        return "MY_CONNECTION_SETTING"
    else:
        # Running locally with Azurite
        return "AzureWebJobsStorage"

# app = func.WsgiFunctionApp(app=flask_app.wsgi_app,  http_auth_level=func.AuthLevel.ANONYMOUS)
app = func.FunctionApp()




@app.function_name("my_content_index")
@app.route(route="index", auth_level=func.AuthLevel.ANONYMOUS)
def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    """Each request is redirected to the WSGI handler.
    """
    return func.WsgiMiddleware(flask_app.wsgi_app).handle(req, context)





@app.route(route="listofids", auth_level=func.AuthLevel.ANONYMOUS)
@app.function_name("listofids")
@app.blob_input(
    arg_name="listofuserids",
    path=blob_path+"test_list_user_ids.joblib",
    connection=get_connection_setting(),
    data_type="binary"
    )
def listofids(req: func.HttpRequest, listofuserids: func.InputStream) :
    """returns the list of user_ids
    """
    # load .joblib 
    list_of_user_ids_array = load(BytesIO(listofuserids.read()))
    # sort user_ids
    list_of_user_ids_array = np.sort(list_of_user_ids_array)
    # as a list
    list_of_user_ids_array = list_of_user_ids_array.tolist()

    return func.HttpResponse(json.dumps({'list' : list_of_user_ids_array}), mimetype="application/json")





@app.route(route="recsfrommodel/{userid:int}/{ncf:int}", auth_level=func.AuthLevel.ANONYMOUS, methods=["get"])
@app.function_name("recsfrommodel")
@app.blob_input(arg_name="data", path=blob_path+"data_light.joblib", connection=get_connection_setting(), data_type="binary")
@app.blob_input(arg_name="model", path=blob_path+"collab_model_light.joblib", connection=get_connection_setting(), data_type="binary")
@app.blob_input(arg_name="emb", path=blob_path+"emb_reduced.joblib", connection=get_connection_setting(), data_type="binary")
@app.blob_input(arg_name="meta", path=blob_path+"meta_cb.joblib", connection=get_connection_setting(), data_type="binary")
def recsfrommodel(req: func.HttpRequest, data: func.InputStream, model: func.InputStream, emb: func.InputStream, meta: func.InputStream) :
    """returns the list of user_ids
    """
    # load .joblib s
    data_loaded = load(BytesIO(data.read()))
    print("data is loaded")
    model_loaded = load(BytesIO(model.read()))
    print("model is loaded")
    emb_loaded = load(BytesIO(emb.read()))
    print("emb is loaded")
    meta_loaded = load(BytesIO(meta.read()))
    print("meta is loaded")
    # extract "user_id"
    # wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
    print("track - func - recsfrommodel/req params", req.route_params)
    print("track - func - recsfrommodel/req params", req.route_params["userid"])
    user_id = int(req.route_params["userid"])
    n_cf = int(req.route_params["ncf"])
    # wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
    print("track - func - recsfrommodel/user_id", user_id)
    print("track - func - recsfrommodel/n_cf", n_cf)
    # top5
    top5 = mf.cf_and_cb_mix_top_5(
        user_id=user_id, 
        n_cf=n_cf,
        data=data_loaded, 
        trained_model=model_loaded,
        emb=emb_loaded,
        meta=meta_loaded
        )

    return func.HttpResponse(json.dumps({'recs' : top5}), mimetype="application/json")





# @app.route(route="recsfromcfmodel/{userid:int}/{ncf:int}", auth_level=func.AuthLevel.ANONYMOUS, methods=["get"])
# @app.function_name("recsfromcfmodel")
# @app.blob_input(arg_name="data", path=blob_path+"data_light.joblib", connection=get_connection_setting(), data_type="binary")
# @app.blob_input(arg_name="model", path=blob_path+"collab_model_light.joblib", connection=get_connection_setting(), data_type="binary")
# def recsfromcfmodel(req: func.HttpRequest, data: func.InputStream, model: func.InputStream) :
#     """returns the list of user_ids
#     """
#     # load .joblib s
#     data_loaded = load(BytesIO(data.read()))
#     print("data is loaded")
#     model_loaded = load(BytesIO(model.read()))
#     print("model is loaded")
#     # extract "user_id"
#     # wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
#     print("track - func - recsfrommodel/req params", req.route_params)
#     print("track - func - recsfrommodel/req params", req.route_params["userid"])
#     user_id = int(req.route_params["userid"])
#     n_cf = int(req.route_params["ncf"])
#     # wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
#     print("track - func - recsfrommodel/user_id", user_id)
#     print("track - func - recsfrommodel/n_cf", n_cf)
#     # sort user_ids
#     top5 = mf.cf_top_5(user_id=user_id, data_cf=data_loaded[["user_id", "article_id", "rating"]], trained_model=model_loaded)

#     return func.HttpResponse(json.dumps({'recscf' : top5}), mimetype="application/json")






# @app.route(route="recsfromcbmodel/{userid:int}/{ncf:int}", auth_level=func.AuthLevel.ANONYMOUS, methods=["get"])
# @app.function_name("recsfromcbmodel")
# @app.blob_input(arg_name="data", path=blob_path+"data_light.joblib", connection=get_connection_setting(), data_type="binary")
# @app.blob_input(arg_name="emb", path=blob_path+"emb_reduced.joblib", connection=get_connection_setting(), data_type="binary")
# @app.blob_input(arg_name="meta", path=blob_path+"meta_cb.joblib", connection=get_connection_setting(), data_type="binary")
# def recsfromcbmodel(req: func.HttpRequest, data: func.InputStream, emb: func.InputStream, meta: func.InputStream) :
#     """returns the list of user_ids
#     """
#     # load .joblib s
#     data_loaded = load(BytesIO(data.read()))
#     print("data is loaded")
#     emb_loaded = load(BytesIO(emb.read()))
#     print("emb is loaded")
#     meta_loaded = load(BytesIO(meta.read()))
#     print("meta is loaded")
#     # extract "user_id"
#     # wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
#     print("track - func - recsfrommodel/req params", req.route_params)
#     print("track - func - recsfrommodel/req params", req.route_params["userid"])
#     user_id = int(req.route_params["userid"])
#     n_cf = int(req.route_params["ncf"])
#     # wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
#     print("track - func - recsfrommodel/user_id", user_id)
#     print("track - func - recsfrommodel/n_cf", n_cf)
#     # sort user_ids
#     top5 = mf.cb_top_5(user_id=user_id, data_cb=data_loaded[["user_id", "click_timestamp", "article_id", "category_id"]], emb=emb_loaded, meta=meta_loaded)

#     return func.HttpResponse(json.dumps({'recscb' : top5}), mimetype="application/json")






@app.function_name("my_content_recommendation_results")
@app.route(route="result", auth_level=func.AuthLevel.ANONYMOUS)
def result(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    """Each request is redirected to the WSGI handler.
    """
    print("track - func - result/req params", req.form)
    return func.WsgiMiddleware(flask_app.wsgi_app).handle(req, context)