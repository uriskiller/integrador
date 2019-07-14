#!flask/bin/python3
#!/bin/sh
from flask import Flask, Response, request, render_template, session, redirect, escape, url_for
from hashlib import md5
import MySQLdb
import cv2
import sys
import time
import pdfkit
import datetime
import random
import numpy as np
import os
import matplotlib.pyplot as plt
from darkflow.net.build import TFNet
from io import BytesIO
import base64
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from PIL import Image
import PIL.Image
from utils import label_map_util
from utils import visualization_utils as vis_util


app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config['PDF_FOLDER'] = 'static/pdf/'
app.config['TEMPLATE_FOLDER'] = 'templates/'
db = MySQLdb.connect(host="localhost", user="root", passwd="12354", db="estacionamiento")
cur = db.cursor()
cur2 = db.cursor()
cur3 = db.cursor()

options = {
        'model': 'cfg/yolo.cfg',
        'load': 'bin/yolov2.weights',
        'threshold': 0.3,
        'gpu': 1.0
    }


tfnet =TFNet(options)


@app.route("/AdminYolo")
def index():
    if 'username' in session:
        username_session = escape(session['username']).capitalize()
        return render_template('upload.html', session_user_name=username_session)
    return render_template('index.html')


@app.route("/Usuario")
def usuario():
    return render_template("usuario.html")

@app.route("/Response",methods=['GET', 'POST'])
def res():
    if request.method == 'POST':
        form_bol  = request.form['txt_boleto']
        query='SELECT numBoleto,tipo,horaEntrada,cuota,seccion,TIMEDIFF(DATE_FORMAT(NOW( ), "%H:%i:%S" ), horaEntrada) tiempoTranscurrido,'
        query+='round(sum(TIME_TO_SEC(TIMEDIFF(DATE_FORMAT(NOW( ), "%H:%i:%S" ),horaEntrada))/3600*cuota)) as cuotaActual '
        query+='FROM estacionamiento.cars where basedate=CURDATE() AND numBoleto="'+form_bol+'" AND token=1;'
        cur.execute(query)
        data=cur.fetchall()
        for row in data:
            b=row[0]
        if b == form_bol:
            return render_template("usuarioPrincipal.html",boleto=form_bol,info=data)
        return render_template("usuario.html",error="Numero invalido")

@app.route('/convert')
def konversi():
    htmlfile = app.config['TEMPLATE_FOLDER'] + 'Pago.html'
    pdffile = app.config['PDF_FOLDER'] + 'demo.pdf'
    pdfkit.from_file(htmlfile, pdffile)
    return redirect("http://localhost:5000/static/pdf/demo.pdf", code=302)

@app.route("/Cuenta")
def cuenta():
    if 'username' in session:
        username_session = escape(session['username']).capitalize()
        query="SELECT email from user where username='%s' " % \
        (username_session)
        cur.execute(query)
        data = cur.fetchone()[0]
        db.commit()
        return render_template("cuenta.html",usuario=username_session,email=data)
    return redirect(url_for('login'))

@app.route("/Editar",methods=['GET', 'POST'])
def editar():
    if 'username' in session:
        username_session = escape(session['username']).capitalize()
        if request.method == 'POST':
            form_pass  = request.form['txt_pass']
            form_email  = request.form['txt_email']
            print(form_pass)
            query="UPDATE user SET password = '%s',email='%s' WHERE (username = '%s');" % \
            (form_pass,form_email,username_session)
            cur.execute(query)
            db.commit()
            return redirect(url_for('cuenta'))
    return redirect(url_for('login'))

@app.route('/',methods=['GET', 'POST'])
def indexw():
    if 'username' in session:
        username_session = escape(session['username']).capitalize()
        opt_param = request.args.get("fil")
        query='SELECT id,numBoleto,tipo,horaEntrada,cuota,seccion, TIMEDIFF(DATE_FORMAT(NOW( ), "%H:%i:%S" ), horaEntrada) tiempoTranscurrido,'
        query+='round(sum(TIME_TO_SEC(TIMEDIFF(DATE_FORMAT(NOW( ), "%H:%i:%S" ),horaEntrada))/3600*cuota)) as cuotaActual '
        query+='FROM estacionamiento.cars where basedate=CURDATE() and token = 1 '

        if request.method == 'POST':
            filtro  = str(request.form['txt_filtro'])
            query+='AND numBoleto LIKE "%'+filtro+'%" group by id;'
        elif opt_param is None:
            filtro=""
            query+='AND tipo LIKE "%'+filtro+'%" group by id;'
        else:
            arg = request.args['fil']
            filtro=arg
            query+='AND tipo LIKE "%'+filtro+'%" group by id;'

        cur.execute(query)
        data = cur.fetchall()
        #query count
        cur2.execute('SELECT count(*) from cars where basedate=CURDATE() and token=1;')
        num= 100 - int(cur2.fetchone()[0])
        return render_template('principal.html', session_user_name=username_session,data=data,num=num)
    return redirect(url_for('login'))


@app.route('/Admin')
def admin():
    if 'username' in session:
        username_session = escape(session['username']).capitalize()
        return render_template('admin.html', session_user_name=username_session)
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if 'username' in session:
        return redirect(url_for('/'))
    if request.method == 'POST':
        username_form  = request.form['username']
        password_form  = request.form['password']
        cur.execute("SELECT COUNT(1) FROM user WHERE Username = %s;", [username_form]) # CHECKS IF USERNAME EXSIST
        if cur.fetchone()[0]:
            cur.execute("SELECT Password FROM user WHERE Username = %s;", [username_form]) # FETCH THE HASHED PASSWORD
            for row in cur.fetchall():
                if password_form == row[0]:
                    session['username'] = request.form['username']
                    return redirect(url_for('indexw'))
                else:
                    error = "Datos incorrectos"
        else:
            error = "Datos incorrectos"
    return render_template('index.html', error=error)

@app.route('/delete',methods=['GET','POST'])
def delete():
    if 'bol' in request.args:
      bol = request.args['bol']
      name_form  = request.form['txt_nombre']
      tarjeta_form  = request.form['txt_tarjeta']
      cur.execute('SELECT round(sum(TIME_TO_SEC(TIMEDIFF(DATE_FORMAT(NOW( ), "%H:%i:%S" ),horaEntrada))/3600*cuota)) as cuotaActual,id FROM cars WHERE numBoleto="'+bol+'";')
      data=cur.fetchall()
      for row in data:
        monto=row[0]
        RID=row[1]
      now = datetime.datetime.now()
      fechapago=now.strftime("%Y-%m-%d %H:%M:%S")
      query2="INSERT INTO payments (monto,fechapago,nombre,tarjeta,RID_Car) VALUES(%d,'%s','%s','%s',%s)" % \
      (monto,fechapago,name_form,tarjeta_form,RID)
      cur2.execute(query2)
      cur3.execute("UPDATE cars set token='0' WHERE numBoleto='"+bol+"'")
      #Insertando      
      db.commit()
      return render_template('Pago.html',status="Boleto pagado.",total=monto,fpago=fechapago)
    if 'id' in request.args:
      id = request.args['id']
      cur.execute("DELETE FROM cars WHERE id="+id)
      db.commit()
      return redirect(url_for('indexw')) 
    

def ins(label):
    tiempo = time.strftime("%H:%M:%S")
    lista = ["A", "B", "C", "Z", "Y", "Z"]
    boleto=random.choice(lista)+str(random.randint(1000,9999))

    cuota=0
    seccion=""
    tipo=""

    if "motorcycle" in label:
      cuota=10
      seccion="M"
      tipo="Motocicleta"
    elif "car" in label:
      cuota=25
      seccion="C"
      tipo="Carro"
    elif "truck" in label:
      cuota=30
      seccion="T"
      tipo="Camioneta"
    else:
      cuota=0
      seccion="No identificado"

    print(tipo,"detectado, generando informacion de boleto...")
    time.sleep(5)  

    query="INSERT INTO cars (numBoleto,tipo,horaEntrada,cuota,seccion,imagen,basedate,token) VALUES ('%s','%s','%s','%d','%s',null,CURDATE(),'1')" % \
    (boleto, tipo , tiempo, cuota, seccion)
    print("Levantando pluma...")
    time.sleep(3)
    print("El vehiculo ha ingresado")
    print("Capturando...")
    cur.execute(query);
    db.commit()



def get_frame2():
    flat=True

    sys.setrecursionlimit(10000)
    # What model to download.
    MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
    NUM_CLASSES = 90


    # ## Download Model
    if not os.path.exists(MODEL_NAME + '/frozen_inference_graph.pb'):
      print ('Downloading the model')
      opener = urllib.request.URLopener()
      opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
      tar_file = tarfile.open(MODEL_FILE)
      for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
          tar_file.extract(file, os.getcwd())
      print ('Download complete')
    else:
      print ('Model already exists')

    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    cap = cv2.VideoCapture(0)

    # Running the tensorflow session
    with detection_graph.as_default():
      with tf.Session(graph=detection_graph) as sess:
       ret = True
       while (ret):
          ret,image_np = cap.read()
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)
          image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
          # Each box represents a part of the image where a particular object was detected.
          boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
          # Each score represent how level of confidence for each of the objects.
          # Score is shown on the result image, together with the class label.
          scores = detection_graph.get_tensor_by_name('detection_scores:0')
          classes = detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = detection_graph.get_tensor_by_name('num_detections:0')
          # Actual detection.
          (boxes, scores, classes, num_detections) = sess.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
          # Visualization of the results of a detection.

          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=8)
    #      plt.figure(figsize=IMAGE_SIZE)
    #      plt.imshow(image_np)
          imgencode=cv2.imencode('.jpg',image_np)[1]
          lb=[category_index.get(value).get('name') for index,value in enumerate(classes[0]) if scores[0,index] > 0.5]
          if "car" in lb and flat==True:
            cap.release()  
            flat=False    
            ins(lb)
            time.sleep(2)
            flat=True
            cap = cv2.VideoCapture(0)
          if "motorcycle" in lb and flat==True:
            cap.release()  
            flat=False   
            ins(lb) 
            time.sleep(2)
            flat=True
            cap = cv2.VideoCapture(0)
          if "truck" in lb and flat==True:
            cap.release()  
            flat=False    
            ins(lb)
            time.sleep(2)
            flat=True
            cap = cv2.VideoCapture(0)

          if cv2.waitKey(25) & 0xFF == ord('q'):
              exit()
          stringData=imgencode.tostring()
          yield (b'--frame\r\n'
              b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')


@app.route('/calc2')
def calc2():
     return Response(get_frame2(),mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route("/upload", methods=["POST"])
def upload():

    target = os.path.join(APP_ROOT, 'images/')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)
    for upload in request.files.getlist("file"):
        print(upload)
        filename = upload.filename
        destination = "/".join([target, filename])
        print("Save it to:", destination)
        upload.save(destination)

    img = cv2.imread(destination, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pred = tfnet.return_predict(img)

    for element in pred:
        element.pop('confidence', None)

    for ele in pred:
        t = (ele['topleft']['x'], ele['topleft']['y'])
        br = (ele['bottomright']['x'], ele['bottomright']['y'])
        label = ele['label']

        img = cv2.rectangle(img, t, br, (0, 255, 0), 7)
        img = cv2.putText(img, label, t, cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 3)
        plt.imshow(img)

    plt.savefig('static/plot_.png')

    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue())
    result = figdata_png

    tiempo = time.strftime("%H:%M:%S")
    lista = ["A", "B", "C", "Z", "Y", "Z"]
    boleto=random.choice(lista)+str(random.randint(1000,9999))

    cuota=0
    seccion=""

    if label == "motorbike":
      cuota=10
      seccion="M"
      label="Motocicleta"
    elif label == "car":
      cuota=25
      seccion="C"
      label="Carro"
    elif label == "truck":
      cuota=30
      seccion="T"
      label="Camioneta"
    else:
      cuota=0
      seccion="No identificado"

    query="INSERT INTO cars (numBoleto,tipo,horaEntrada,cuota,seccion,imagen,basedate,token) VALUES ('%s','%s','%s','%d','%s',null,CURDATE(),'1')" % \
    (boleto, label , tiempo, cuota, seccion)
    print(query)
    cur.execute(query);
    db.commit()


    return render_template('complete.html', variable=label, pred=pred, tiempo=tiempo, bol=boleto, cuota=cuota, seccion=seccion)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return render_template('index.html')

app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'
usuario="Undefined"

if __name__ == "__main__":
    app.run(port=5000,debug=True,use_reloader=True)
