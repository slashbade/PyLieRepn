import execjs
import json
#from tkinter import *

def w2DominoTableau(w):
    node = execjs.get()
    ctx2 = node.compile(open('lieToolbox/Tableau.js',encoding='utf-8').read())
    strw = [str(i) for i in w]
    ent = ' '.join(strw)
    rst = ctx2.call("calc", ent)
    return rst

def vertical_domino_boxes(domino):
    domino_list = domino['dominoList']
    count = 0
    for dmn in domino_list:
        if dmn['horizontal'] == False:
            count += 1
    return count

# def printDominoTableau(domino):
#     base = Tk()
#     tableau_cv = Canvas(base, height=10)
#     d = 25
#     x0 = 20
#     y0 = 20
#     tableau_cv.config(height=60 + d * domino['dominoGrid']['columnLengths'][0], width=100 + d * domino['dominoGrid']['rowLengths'][0])
#     domino_list = domino['dominoList']
#     for dmn in domino_list:
#         if dmn['horizontal']:
#             dmn_wd = 2*d
#             dmn_ht = d
#         else:
#             dmn_wd = d
#             dmn_ht = 2*d
#         tableau_cv.create_rectangle(x0+d*dmn['x'], 
#                                     y0+d*dmn['y'], 
#                                     x0+d*dmn['x']+dmn_wd, 
#                                     y0+d*dmn['y']+dmn_ht)
#         tableau_cv.create_text(x0+d*dmn['x']+dmn_wd/2, 
#                                y0+d*dmn['y']+dmn_ht/2, 
#                                text=str(dmn['n']))
#     tableau_cv.pack()
#     base.mainloop()
    
if __name__ == '__main__':
    domino = w2DominoTableau([-3,1,-2,5,-4])
    #printDominoTableau(domino)
    print(vertical_domino_boxes(domino))
    # ent = "1 2 -4 3 -5 -6"
    # rst = ctx2.call("calc",ent)
    # print(rst["left"]["dominoList"])
    # data = json.dumps(rst, indent=4,ensure_ascii=False, sort_keys=False,separators=(',', ':'))
    # print(data)
    


