from flask import Flask,render_template,request
from model_prediction import Prediction

app=Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict",methods=["GET","POST"])
def predict():
    if request.method=="POST":
        online_order=request.form.get("online_order")
        book_table=request.form.get("book_table")
        votes=request.form.get("votes")
        rest_type=request.form.get("rest_type")
        cost=request.form.get("cost")
        type=request.form.get("type")
        location=request.form.get("location")
        model_prediction_obj=Prediction(
            online=online_order,
            reservations=book_table,
            votes=votes,
            location=location,
            rest_type=rest_type,
            cost_for_two=cost,
            type=type)
        model_path=r"D:\projects\Zomato_Rating_Prediction\artifacts\model.pkl"
        preprocessor_path=r"D:\projects\Zomato_Rating_Prediction\artifacts\Preprocessor.pkl"
        output=model_prediction_obj.Predict_rating(model_path=model_path,preprocessor_path=preprocessor_path)
        def rating_star(rate):
            if rate==2:
                return "⭐⭐"
            elif rate==3:
                return "⭐⭐⭐"
            elif rate==4:
                return "⭐⭐⭐⭐"
            elif rate==5:
                return "⭐⭐⭐⭐⭐"
            else:
                return "⭐"
        
        pred=rating_star(output)
        return render_template("home.html",pred=pred)
    else:
        return render_template("index.html")


if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)