# encoding=utf-8
from flask import Flask, request, make_response,jsonify
from flask_uploads import UploadSet, configure_uploads
from flask import render_template
import os
import base64
from ImageTools import colorcontrast,ImageSearch
from io import BytesIO
from PIL import Image
import json

app = Flask(__name__)

app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_PATH'] = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOADS_DEFAULT_DEST'] = os.path.dirname(os.path.abspath(__file__))

uploaded_photos = UploadSet()
configure_uploads(app, uploaded_photos)

handling_filename=""

@app.route('/go/')
def go_other():
    return render_template("PicSlide.html")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def flask_upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'code': -1, 'filename': '', 'msg': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'code': -1, 'filename': '', 'msg': 'No selected file'})
        else:
            try:
                print(file)
                filename = uploaded_photos.save(file)
                print(filename)
                global handling_filename
                handling_filename=filename
                return jsonify({'code': 0, 'filename': filename, 'msg': uploaded_photos.url(filename)})
            except Exception as e:
                return jsonify({'code': -1, 'filename': '', 'msg': 'Error occurred'})
    else:
        return jsonify({'code': -1, 'filename': '', 'msg': 'Method not allowed'})


@app.route('/files/<string:filename>', methods=['GET','POST'])
def show_photo(filename):
    if request.method == 'GET':
        if filename is None:
            pass
        else:
            image_data = base64.b64encode(open(os.path.join(app.config['UPLOAD_PATH'], 'files/%s' % filename), "rb").read())
            response = make_response(image_data)
            response.headers['Content-Type'] = 'image/png'
            return response
    else:
        pass

@app.route('/simimg/<string:filename>',methods=['GET','POST'])
def show_similar_photoes(filename):
    if request.method=='GET':
        s=ImageSearch.SearchImage()
        t=s.GetImageFeature(os.path.join(app.config['UPLOAD_PATH'], 'files/%s' % filename))
        buffered = BytesIO()
        img_strs={}
        for idx,i in enumerate(t):
            img=Image.open(i)
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            key='img'+str(idx)
            img_strs[key]=img_str
        jsonobj=json.dumps(img_strs)
        return jsonobj
    else:
        return 'fail'

@app.route('/ImageTools/<string:method>',methods=['GET','POST'])
def change_contrast(method):
    radio = request.form[method]
    if method=="bilfilter":
        radio2=request.form['another']
    img_path = os.path.join(app.config['UPLOAD_PATH'], 'files/%s' % handling_filename)
    basic_process = colorcontrast.ColorContrast(img_path)
    buffered = BytesIO()
    if method=="contrast":
        img = basic_process.change_contrast(float(radio))
    elif method=="light":
        img=basic_process.change_bright(float(radio))
    elif method=="saturation":
        img=basic_process.change_saturation(float(radio))
    elif method=="histeq":
        img=basic_process.histeq()
    elif method=="avefilter":
        img=basic_process.average_filter()
    elif method=="midfilter":
        img=basic_process.median_filter()
    elif method=="guafilter":
        img=basic_process.Guassion_filter(float(radio))
    elif method=="bilfilter":
        img=basic_process.bilateral_filter(float(radio),float(radio2))
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    response = make_response(img_str)
    response.headers['Content-Type'] = 'image/png'
    return response

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9000, debug=True)