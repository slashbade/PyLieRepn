from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)
from werkzeug.exceptions import abort

from re import split
from LieToolbox.Repn.weight import Weight, HighestWeightModule
from LieToolbox.Repn.GK_dimension import antidominant
import json
import numpy as np

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
def GKdim_get():
    if request.method == 'POST':
        input_str = request.form['weight']
        weight = np.array(list(map(eval, split(', |,|，| ', input_str))))
        rank = eval(request.form['rank'])
        typ = request.form['lieType']
        from LieToolbox.Repn.GK_dimension import GK_dimension
        gkdim, info = GK_dimension(typ, rank, weight)
        return render_template('lie/GKdim.html', gkdim=gkdim, info=info)
    else:
        return render_template('lie/GKdim.html')

@bp.route('/lie/tableau', methods=('GET', 'POST'))
def tableau():
    if request.method == 'POST':
        # Get the list of floats from the form input
        entryStr = request.form['weight']
        lbd = Weight.parseStrWeight(entryStr, 'A')
        pt = lbd.constructTableau()

        return render_template('lie/tableau.html', tableau_data=pt.entry)

    return render_template('lie/tableau_input.html')

@bp.route('/lie/antidominant', methods=('GET', 'POST'))
def antidominant_get():
    if request.method == 'POST':
        # This support only real weights
        input_str = request.form['weight']
        weight = np.array(list(map(eval, split(', |,|，| ', input_str))))
        rank = len(weight)
        typ = request.form['lieType']
        antidominant_weyl, antidominant_weight = antidominant(typ=typ, rank=rank, weight_=weight)
        
        return render_template('lie/antidominant.html', 
                               antidominant_weyl=str(antidominant_weyl), 
                               antidominant_weight=str(np.round(antidominant_weight, 3)))
    else:
        return render_template('lie/antidominant.html')
    