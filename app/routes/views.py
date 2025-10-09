from flask import Blueprint, current_app, render_template

bp = Blueprint("views", __name__)


@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/wall-art")
def wall_art():
    return render_template("wall_art/index.html")
