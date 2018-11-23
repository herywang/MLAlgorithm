import json
import requests
#通过get方法发送请求
def requestsTest1():
	r = requests.get("http://jwxt.hbuas.edu.cn/jsxsd/xsxk/xsxk_index?jx0502zbid=5FAF2F7FF31F43B3BFB896BFD1D9FF17")
	content = r.text
	print(content)

def requseTest2():
	myparam = {'wd':'Linux'}
	r = requests.get('https://www.baidu.com/s',params = myparam)
	print(r.url)
	print (r.text,end="")

#无限注册外挂
def request_post():
	mydata = {'userAccount':'2016117119','userPassword':'13797763577wh'}
	r = requests.post('http://jwxt.hbuas.edu.cn/jsxsd/',data = mydata)
	print(r.text)

#传输json格式的数据
def request_json():
	mydata = {'username':'683602165','password':'123142342432'}
	r = requests.post('http://jwxt.hbuas.edu.cn/jsxsd/xsxk',data = json.dumps(mydata))
	print (r.text)

#上传文件
def request_files():
	myfile = {'file':open('106.jpg','rb')}
	r = requests.post('http://httpbin.org/post',files = myfile)
	print (r.text)

request_post()

