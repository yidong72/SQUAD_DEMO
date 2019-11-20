import cherrypy
from tr_infer import Model
m = Model()

cherrypy.config.update({'server.socket_port': 8888, 'server.socket_host': '0.0.0.0'})

class HelloWorld(object):

    @cherrypy.expose
    def index(self):
        return "Hello world!"

    @cherrypy.expose
    def infer(self, para="empty", question="empty"):
        return m.inference(para, question)


if __name__ == '__main__':
    cherrypy.quickstart(HelloWorld())
