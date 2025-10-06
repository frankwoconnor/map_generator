from flask import Blueprint, render_template, current_app

bp = Blueprint('views', __name__)

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/wall-art')
def wall_art():
    return render_template('wall_art/index.html')
