function drawTable(canvas, p, borderColor, faceColor) {
    if (canvas===null) {}
    else {
        if (canvas.getContext) {
            var ctx = canvas.getContext("2d");
            let d = 20;
            let x0 = 0;
            let y0 = 0;
            width = p[0] * d
            height = p.length * d
            canvas.setAttribute("width", width)
            canvas.setAttribute("height", height)
            var x = x0;
            var y = y0;
            
            ctx.strokeStyle = borderColor;
            ctx.fillStyle = faceColor;
            for (var i = 0; i < p.length; i++) {
                x = x0;
                for (var j = 0; j < p[i]; j++) {
                    ctx.fillRect(x, y, d, d);
                    ctx.strokeRect (x, y, d, d);
                    x += d;
                }
                y += d;
            }
        }
        else {
            alert("cannot get context")
        }
    }
}

function drawDominoTable(canvas, domino, borderColor, faceColor) {
    if (canvas===null){}
    else{
        if (canvas.getContext) {
            var ctx = canvas.getContext("2d");
            let d = 20;
            let x0 = 0;
            let y0 = 0;
            var x = x0;
            var y = y0;
            width = domino['dominoGrid']['columnLengths']['length'] * d
            height = domino['dominoGrid']['rowLengths']['length'] * d
            canvas.setAttribute("width", width)
            canvas.setAttribute("height", height)
            dominoList = domino['dominoList']
            console.log(domino)
            console.log(width)
            ctx.strokeStyle = borderColor;
            ctx.fillStyle = faceColor;
            for (var i = 0; i < dominoList.length; i++) { 
                if (dominoList[i]['horizontal']) {
                    var dmnWd = 2 * d;
                    var dmnHt = d;
                }
                else {
                    var dmnWd = d;
                    var dmnHt = 2 * d;
                }
                ctx.fillRect(x0 + d * dominoList[i]['x'], y0 + d * dominoList[i]['y'], dmnWd, dmnHt)
                ctx.strokeRect(x0 + d * dominoList[i]['x'], y0 + d * dominoList[i]['y'], dmnWd, dmnHt)
            }
        }
    }
}
