from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("forest_fire.html")


@app.route("/predict", methods=["POST", "GET"])
def predict():
    pipeline = PredictPipeline()
    try:
        int_features = [[int(x) for x in request.form.values()]]
    except:
        return render_template(
            "forest_fire.html",
            error="Input values are less",
        )

    if any(feature < 0 or feature > 100 for feature in int_features[0]):
        return render_template(
            "forest_fire.html",
            error="Invalid Input, values must be in range from [0, 100]",
        )
    try:
        prediction = pipeline.predict(int_features)
        print(prediction)
        output = "{0:.{1}f}".format(prediction[0][1], 2)
    except Exception as e:
        print(e)
        render_template("forest_fire.html", pred="There seems some error")
    if output > str(0.5):
        return render_template(
            "forest_fire.html",
            pred=f"Your Forest is in Danger.\nProbability of fire occuring is {output}",
            error="",
        )

    return render_template(
        "forest_fire.html",
        pred=f"Your Forest is safe.\n Probability of fire occuring is {output}",
        error="",
    )


if __name__ == "__main__":
    app.run(debug=True)
