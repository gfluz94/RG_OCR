from flask import Flask, render_template, url_for, redirect
from forms import AddImage, RGFields

from ocr_rg import ImageDewarper, RGReader
import cv2


app = Flask(__name__)
app.config["SECRET_KEY"] = "mysecretkey"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/add_image", methods=["GET", "POST"])
def add_image():

    form = AddImage()
    results = RGFields()

    if form.validate_on_submit():

        rg = form.rg.data
        dewarper = ImageDewarper(blur_ksize=5, threshold_value=195, dilation_ksize=5, output_size=600)
        rg_reader = RGReader(dewarper)
        output_rg = rg_reader.read_img(rg)
        
        results.rg.data = output_rg["RG"]
        results.exped.data = output_rg["DATA_EXPED"]
        results.name.data = output_rg["NOME"]
        results.mother.data = output_rg["NOME_MAE"]
        results.father.data = output_rg["NOME_PAI"]
        results.bdate.data = output_rg["DATA_NASC"]
        results.cpf.data = output_rg["CPF"]
        results.city.data = output_rg["CIDADE_ORIGEM"]
        results.state.data = output_rg["UF_ORIGEM"]

        return render_template("data.html", form=results)

    return render_template("add.html", form=form)

@app.errorhandler(404)
def error404(e):
    return render_template("404.html"), 404


if __name__ == "__main__":
    app.run(debug=True)



    