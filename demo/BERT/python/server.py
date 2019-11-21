from multiprocessing import Process, Queue

input_queue = Queue()
output_queue = Queue()

def run_server(input_queue, output_queue):
    import cherrypy

    cherrypy.config.update({'server.socket_port': 8888,
#        'environment': 'production',
        'engine.autoreload.on': False,
#        'server.thread_pool':  1,
        'server.socket_host': '0.0.0.0'})

    class HelloWorld(object):
        import cherrypy
        @cherrypy.expose
        def index(self):
         return """<html>
              <head>

    <script>
    function clicked() {
      var xhttp = new XMLHttpRequest();
      xhttp.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
         console.log(this.responseText);
         var jsonResponse = JSON.parse(this.responseText);
         document.getElementById("answer").value = jsonResponse['result']
         document.getElementById("probability").value = jsonResponse['p']
        }
      };

      xhttp.open("POST", "infer", true);
      xhttp.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
      var para_doc = document.getElementById("para").value;
      var question_doc = document.getElementById("question").value;
      //console.log(para_doc);
      //console.log(question_doc);
      xhttp.send(JSON.stringify({ "para": para_doc, "question": question_doc }));
    }
    </script>

              </head>
              <body>
              <div style="width:800px; margin:0 auto;">
    <textarea rows="50" cols="50" id="para">
    You may elect to defer a percentage of your eligible compensation into the Plan after you satisfy the Plan's eligibility requirements. The percentage of your eligible compensation you elect will be withheld from each payroll and contributed to an Account in the Plan on your behalf. For pre-tax contributions being withheld from your compensation, the percentage you defer is subject to an annual limit of the lesser of 80.00% of eligible compensation or $19,000 (in 2019; thereafter as adjusted by the Secretary of the Treasury) in a calendar year.
    </textarea><br>
    Question:<input type="text" id="question" value="What is the annual limit of percentage you defer" size=50><br>
    <button type="button" onclick="clicked()">Send</button><br>
    Answer:<input type="text" id="answer" size=50 disabled=true><br>
    Probability:<input type="text" id="probability" disabled=true size=10>
    </div>
    </body>
            </html>"""

        @cherrypy.expose
        @cherrypy.tools.json_out()
        @cherrypy.tools.json_in()
        def infer(self):
            input_json = cherrypy.request.json
            input_queue.put((input_json['para'], input_json['question']))
            return output_queue.get()
    cherrypy.quickstart(HelloWorld())

if __name__ == '__main__':
    p = Process(target=run_server, args=(input_queue, output_queue))
    p.start()

    from tr_infer import Model
    m = Model()
    while True:
        inputs = input_queue.get()
        r = m.inference(inputs[0], inputs[1])
        output_queue.put(r)
