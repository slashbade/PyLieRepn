from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)
from werkzeug.exceptions import abort

from re import split
from lieToolbox.weight import Weight, HighestWeightModule
import json

bp = Blueprint('lie', __name__, url_prefix='/')

@bp.route('/')
def index():
    return render_template('lie/index.html')


@bp.route('/lie/classification', methods=('GET', 'POST'))
def classification():
    if request.method == 'POST':
        entryStr = request.form['weight']
        error = None
        if not entryStr:
            error = 'Weight is required'
        else:
            lieType = request.form['lieType']
            lbd = Weight.parseStrWeight(entryStr, lieType)
            L_lbd = HighestWeightModule(lbd)
            obt = L_lbd.nilpotentOrbit()
            obtInfo = L_lbd.nilpotentOrbitInfo()
            gkdim = L_lbd.GKdim()
            obtInfo['GKdim'] = gkdim
            obtInfo['GKdimInfo'] = L_lbd.GKdimInfo()
            
        if error is None:
            return render_template('lie/classification.html', obtInfo=obtInfo, obtInfojs=json.dumps(obtInfo))
        flash(error)
    return render_template('lie/classification.html')

@bp.route('/lie/GKdim', methods=('GET', 'POST'))


@bp.route('/lie/tableau', methods=('GET', 'POST'))
def tableau():
    if request.method == 'POST':
        # Get the list of floats from the form input
        entryStr = request.form['weight']
        lbd = Weight.parseStrWeight(entryStr, 'A')
        pt = lbd.constructTableau()

        return render_template('lie/tableau.html', tableau_data=pt.entry)

    return render_template('lie/tableau_input.html')
    