function draw() {
    function drawTable(canvas, p) {
        if (canvas===null) {}
        else {
            if (canvas.getContext) {
                var ctx = canvas.getContext("2d");

                let d = 20;
                let x0 = 0;
                let y0 = 0;

                width = p[0] * d
                console.log(Math.max(p))
                height = p.length * d
                canvas.setAttribute("width", width)
                canvas.setAttribute("height", height)
                var x = x0;
                var y = y0;
                ctx.fillStyle = "rgb(200,0,0)";
                for (var i = 0; i < p.length; i++) {
                    x = x0;
                    for (var j = 0; j < p[i]; j++) {
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
    
    
    function drawDominoTable(canvas, domino) {
        if (canvas===null){}
        else{
            if (canvas.getContext) {
                
                var ctx = canvas.getContext("2d");
                let d = 20;
                let x0 = 0;
                let y0 = 0;
                var x = x0;
                var y = y0;
                ctx.fillStyle = "rgb(200,0,0)";
                dominoList = domino['dominoList']
                console.log(dominoList[0])
                for (var i = 0; i < dominoList.length; i++) {
                    
                    if (dominoList[i]['horizontal']) {
                        var dmnWd = 2 * d;
                        var dmnHt = d;
                    }
                    else {
                        var dmnWd = d;
                        var dmnHt = 2 * d;
                    }
                    ctx.strokeRect(x0 + d * dominoList[i]['x'], y0 + d * dominoList[i]['y'], dmnWd, dmnHt)
                }
            }
        }
        

    }

    

    {% if obtInfo %}
        var u = {{obtInfojs | safe}}
        var p = JSON.parse('{{obtEntry}}');
        var canvas = document.getElementById("canvas");
        drawTable(canvas, p)
        
        {% if obtInfo['lieType'] == 'A'%}
            let tableNum = JSON.parse('{{obtInfo['UnitList'] | length}}')
            for (var i = 0; i < tableNum; i++) {
                var p = u['UnitList'][i]['Partition']
                var canvas = document.getElementById("table" + String(i+1))
                drawTable(canvas, p)
            }
        
        {% else %}
        
            var p = u['Integral']['Partition2']
            if (p.length > 0){
                var canvas = document.getElementById("table11")
                drawTable(canvas, p)
            }
            

            var p = u['HIntegral']['Partition2']
            if (p.length > 0) {
                var canvas = document.getElementById("table12")
                drawTable(canvas, p)
            }
            
            
            let tableNum = JSON.parse('{{obtInfo['NHIntegral'] | length}}')
            for (var i = 0; i < tableNum; i++) {
                var p = u['NHIntegral'][i]['Partition']
                var canvas = document.getElementById("table" + String(i+1))
                drawTable(canvas, p)
            }
            
            {% if obtInfo['lieType'] == 'D' and obtInfo['isVeryEven'] == True %}
            
            var rst = u['veryEvenTypeInfo']['Integral']['DominoTableau'];
            var canvas = document.getElementById("tabled1");
            drawDominoTable(canvas, rst)

            var rst = u['veryEvenTypeInfo']['HIntegral']['DominoTableau'];
            var canvas = document.getElementById("tabled2");
            drawDominoTable(canvas, rst)
            {% endif %}
        {% endif %}
    {% endif %}
}