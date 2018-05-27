import json
import base64
from PIL import Image
from io import BytesIO
buffer=BytesIO()
Image.open("/home/wlj/PycharmProjects/FlaskHomework/static/6.jpeg").save(buffer,format='PNG')
a={}
a['img']=base64.b64encode(buffer.getvalue()).decode('utf-8')
a['img2']=base64.b64encode(buffer.getvalue()).decode('utf-8')
b=json.dumps(a)
print(b)