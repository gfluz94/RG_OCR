from flask_wtf import FlaskForm
from wtforms import SubmitField, FileField, StringField
from wtforms.validators import DataRequired

class AddImage(FlaskForm):

    rg = FileField("Please submit a picture of your ID document: ", validators=[DataRequired()])
    submit = SubmitField("Get RG Info")

class RGFields(FlaskForm):

    rg = StringField("RG Number: ")
    exped = StringField("Expedition Date: ")
    cpf = StringField("CPF Number: ")
    name = StringField("Name: ")
    mother = StringField("Mother's Name: ")
    father = StringField("Father's Name: ")
    bdate = StringField("Birth Date: ")
    city = StringField("Hometown: ")
    state = StringField("State: ")
    