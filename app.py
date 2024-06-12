from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("mental_health.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features = [float(x) for x in request.form.values(  )]
    final = [np.array(int_features)]
    print(int_features)
    print(final)
    prediction = model.predict_proba(final)
    # output = '{0:.2f}'.format(prediction[0][1])
    output='{0:.{1}f}'.format(prediction[0][1], 2)
    resources = [
        {"name": "National Suicide Prevention Lifeline", "phone": "9152987821", "website": "https://icallhelpline.org/"},
        {"name": "Crisis Text Line", "phone": " 741741", "website": "https://www.crisistextline.org/"},
        {"name": "NIMHANS", "phone": " +91-80-26995000.", "website": "https://www.nimhans.ac.in/"},
        {"name": "Manastha","phone": " 074286 99696", "website": "https://www.manastha.com/"},
        {"name": "Headspace","phone": " 1800 650 890", "website": "https://www.headspace.com/"},
        {"name": "Verywell Mind", "phone": " (212) 204-4000","website": "https://www.verywellmind.com/"},
        {"name": "The Jed Foundation", "phone": " (212) 647-7544 ","website": "https://www.jedfoundation.org/"},
        {"name": "The American Foundation for Suicide Prevention", "phone": " 1-800-273-8255","website": "https://afsp.org/"}
    ]
    if output>str(0.5):
        return render_template('mental_health.html',pred="Please consider reaching out to a healthcare professional for a thorough evaluation.\n Probability of seeking treatment is {}".format(output),resources=resources)
    else:
        return render_template('mental_health.html',pred="I can see that you've been handling things really well. Don't worry you're Safe!\n Probability of seeking treatment is {}".format(output))


if __name__ == '__main__':
    app.run(debug=True)
