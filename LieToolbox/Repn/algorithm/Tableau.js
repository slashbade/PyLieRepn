/**
 MIT License

 Copyright (c) 2016-2019 Devra Garfinkle Johnson
 Copyright (c) 2016 Christian Johnson

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 */

/**
 * @file
 * This file stores the classes needed to perform the Domino Robinson-Schensted algorithm
 * (and its inverse) and to draw the results.<br>
 * The main class is the {@link Tableau} class, which handles domino tableaux.<br>
 * The {@link Domino} class stores the information about the contents, location,
 * and orientation of one domino in a domino tableau.<br>
 * The {@link DominoGrid} class is a helper class for the {@link Tableau} class.<br>
 * The {@link TableauPair} class holds a pair of {@link Tableau}.<br>
 * The {@link ParameterDominoRS} class handles the input to the
 * Domino Robinson-Schensted algorithm, that is, a signed permutation.<br>
 * {@link TableauRendererDOM} and {@link TableauPairRendererDOM} are the drawing classes.<br>
 * The {@link TableauWithGrid} class is an extension of the {@link Tableau} class.
 * Domino tableaux are used for simple Lie groups of types B, C, and D.
 * There is one grid for each type.<br>
 * @copyright 2016-2018 Devra Garfinkle Johnson, 2016 Christian Johnson
 */

"use strict";

/**
 * This class stores the x and y coordinates of a location
 * (usually in or near a tableau.)
 */
class Square {
        /**
         * @param {Object} table
         * @param {number} table.x
         * @param {number} table.y
         */
        constructor(table) {
                /**
                 * @type {number}
                 */
                this.x = table.x;
                /**
                 * @type {number}
                 */
                this.y = table.y;
        }

        /**
         * makes a copy of the Square
         * @return {Square}
         */
        clone() {
                return new Square(this);
        }

        /**
         * @param {Object} other - the Square (hopefully)
         * to compare to this Square
         * @return {boolean} true if other is a Square
         * whose x abd y coordinates agree with this Square's
         * coordinates
         */
        equals(other) {
                if (!(other instanceof Square)) {
                        return false;
                }

                return this.x == other.x && this.y == other.y;
        }

        /**
         * String representation of the Square
         * @return {string}
         */
        toString() {
                let squareString = "Square { x: " + this.x + ", y: " + this.y + " }";
                return squareString;
        }
}

/**
 * This is a helper class for the {@link Tableau} class.
 * It is one of the member variables of that class.
 */
class DominoGrid {
        /**
         * The constructor populates the DominoGrid from an Array of Dominos.
         * @param {Domino[]} dominoList - an array of Dominos (which is often empty).
         */
        constructor(dominoList) {
                /**
                 * This is a 2D jagged array with the same shape as the tableau
                 * of which it is a member.
                 * Each entry of the array stores a reference to the Domino
                 * in that position of the tableau.
                 * @type {Domino[][]}
                 */
                this.grid = [];
                /**
                 * This array stores the length of each row of the tableau.
                 * @type {number[]}
                 */
                this.rowLengths = [];
                /**
                 * This array stores the length of each column of the tableau.
                 * @type {number[]}
                 */
                this.columnLengths = [];
                dominoList.forEach((domino) => { this.addDomino(domino);});
        }

        /**
         * This function sets the grid location at position (x, y) to
         * the <code>domino</code>.
         * @param {number} x
         * @param {number} y
         * @param {Domino} domino
         */
        set(x, y, domino) {
                if (!this.columnLengths[x] || this.columnLengths[x] <= y) {
                        this.columnLengths[x] = y + 1;
                }

                if (!this.rowLengths[y] || this.rowLengths[y] <= x) {
                        this.rowLengths[y] = x + 1;
                }

                if (!this.grid[y]) {
                        this.grid[y] = [];
                }

                this.grid[y][x] = domino;
        }

        // /**
        //  * This function removes position (x, y) from the grid.
        //  * It requires that the position is at the edge of the tableau.
        //  * @param {number} x
        //  * @param {number} y
        //  */
        // unset(x, y) {
        //         if (x == 0) {
        //                 this.grid.pop();
        //         } else {
        //                 this.grid[y].pop();
        //         }
        //
        //         this.rowLengths[y]--;
        //         this.columnLengths[x]--;
        // }

        /**
         * This function removes position (x, y) from the grid.
         * It requires that the position is at the edge of the tableau.
         * @param {number} x
         * @param {number} y
         * @throws {Error} if the position is not at the edge of the tableau
         */
        unset(x, y) {
                if (this.rowLengths[y] != x + 1 || this.columnLengths[x] != y + 1) {
                        throw new Error("Bad input to unset.");
                }

                if (x == 0) {
                        this.grid.pop();
                        this.rowLengths.pop();
                } else {
                        this.grid[y].pop();
                        this.rowLengths[y]--;
                }

                if (y == 0) {
                        this.columnLengths.pop();
                } else {
                        this.columnLengths[x]--;
                }
        }

        /**
         * This function gets the domino at grid location (x, y).
         * @param {number} x
         * @param {number} y
         * @return {Domino|undefined}
         */
        get(x, y) {
                if (!this.rowLengths[y]) {
                        return undefined;
                }

                return this.grid[y][x];
        }

        /**
         * This function gets the content the domino at grid location (x, y).
         * This content may be a number, a sign, or an empty string.
         * If there is no domino at this location,
         * the function returns undefined.
         * @param {number} x
         * @param {number} y
         * @return {number|string|undefined}
         */
        getContent(x, y) {
                let domino = this.get(x, y);
                if (domino) {
                        return domino.n;
                }

                return undefined;
        }

        /**
         * This function sets the grid locations in the position
         * covered by a domino to that domino.
         * @param {Domino} domino
         */
        addDomino(domino) {
                this.set(domino.x, domino.y, domino);
                if (domino.zero) {
                        return;
                }

                if (domino.horizontal) {
                        this.set(domino.x + 1, domino.y, domino);
                } else {
                        this.set(domino.x, domino.y + 1, domino);
                }

                if (domino.box) {
                        this.set(domino.x + 1, domino.y, domino);
                        this.set(domino.x + 1, domino.y + 1, domino);
                }
        }

        /**
         * This function removes a domino shape from the grid,
         * given that the shape is in a position at the edge of the tableau,
         * that is, given that removing the shape would still leave
         * the shape of a legitimate tableau.
         * @param {Object} position
         * @param {number} position.x
         * @param {number} position.y
         * @param {boolean} position.horizontal
         */
        removeExtremal(position) {
                if (position.horizontal) {
                        if (position.x == 0) {
                                this.grid.pop();
                        } else {
                                this.grid[position.y].pop();
                                this.grid[position.y].pop();
                        }

                        this.rowLengths[position.y] -= 2;
                        this.columnLengths[position.x]--;
                        this.columnLengths[position.x + 1]--;
                } else { // !position.horizontal
                        if (position.x == 0) {
                                this.grid.pop();
                                this.grid.pop();
                        } else {
                                this.grid[position.y + 1].pop();
                                this.grid[position.y].pop();
                        }

                        this.rowLengths[position.y]--;
                        this.rowLengths[position.y + 1]--;
                        this.columnLengths[position.x] -=2;
                }
        }

        /**
         * @param {number} i - the zero-based index of the row
         * @return {number} the length of the ith row
         */
        getRowLength(i) {
                if (!this.rowLengths[i]) {
                        return 0;
                }

                return this.rowLengths[i];
        }

        /**
         * @param {number} j - the zero-based index of the column
         * @return {number} the length of the jth column
         */
        getColumnLength(j) {
                if (!this.columnLengths[j]) {
                        return 0;
                }

                return this.columnLengths[j];
        }
}

/**
 * This class stores the information for one domino of the tableau.
 * Some objects in the Domino class are not actually domino shape,
 * though they do reside in Domino tableaux.  B type number tableaux require
 * a square with a zero in it in the upper-left corner.  Also, sign tableaux
 * have 2x2 boxes in them as well as dominos.
 */
class Domino {
        /**
         * @param {Object} table
         * @param {number|string} table.n - The number occupying the domino.  Though,
         * in some uses, the domino is occupied by a sign, not a number.  In some cases
         * the domino is blank, that is, <code>table.n == ''</code>.
         * @param {number} table.x - The x-coordinate of the left-most square of the domino in the tableau.
         * Zero-based, zero on the left.
         * @param {number} table.y - The y-coordinate of the highest square of the domino in the tableau.
         * Zero-based, zero on top.
         * @param {boolean} [table.horizontal] - If true, the domino is horizontal,
         * if false, the domino is vertical.
         * @param {boolean} [table.box] - Some members of the Domino class are not actually dominos.
         * If table.box is true, then this domino is a 2x2 box.
         * @param {boolean} [table.zero] - If table.zero is true, this "domino" is actually a 1x1 square,
         * holding a 0, situated in the top-left corner of the tableau.
         * @description Exactly one of the optional parameters will be present.
         */
        constructor(table) {
                /**
                 * The content of the Domino
                 * @type {number|string}
                 */
                this.n = table.n;
                /**
                 * The x-coordinate of the left-most square of the domino
                 * in the tableau.
                 * Zero-based, zero on the left.
                 * @type {number}
                 */
                this.x = table.x;
                /**
                 * The y-coordinate of the highest square of the domino
                 * in the tableau.
                 * Zero-based, zero on top.
                 * @type {number}
                 */
                this.y = table.y;
                if (table.box) {
                        /**
                         * If true, then this "domino" is a 2x2 box.
                         * @type {boolean|undefined}
                         */
                        this.box = true;
                } else if (table.zero) {
                        /**
                         * If true, this "domino" is actually a
                         * 1x1 square, holding a 0,
                         * situated in the top-left corner of the tableau.
                         * @type {boolean|undefined}
                         */
                        this.zero = true;
                } else {
                        /**
                         * If true, the domino is horizontal,
                         * if false, the domino is vertical.
                         * @type {boolean|undefined}
                         */
                        this.horizontal = table.horizontal;
                }
        }

        /**
         * makes a copy
         * @return {Domino}
         */
        clone() {
                return new Domino(this);
        }

        /**
         * @return {string}
         */
        toString() {
                let dominoString =  "Domino { n: " + this.n + ", x: " + this.x + ", y: " + this.y + ", ";
                if (this.box) {
                        dominoString += "box: true";
                } else if (this.zero) {
                        dominoString += "zero: true";
                } else {
                        dominoString += "horizontal: " + this.horizontal;
                }

                dominoString += " }";

                return dominoString;
        }

        /**
         * Creates a zero "domino", that is, a 1x1 square,
         * holding a 0, situated in the top-left corner of the tableau.
         * @return {Domino}
         */
        static makeZero() {
                return new Domino({n: 0, x: 0, y: 0, zero: true});
        }

        /**
         * makes a deep copy of an array of Domino
         * @param {Domino[]} dominoList
         * @return {Domino[]}
         */
        static cloneList(dominoList) {
                let newList = [];
                dominoList.forEach((domino) => {newList.push(domino.clone())});
                return newList;
        }
}

/**
 * This class handles domino tableaux, that is, tableaux for types B, C, and D.
 * It stores the same information
 * in two member variables.  So, inserts and deletes have to update both.
 * In addition to the obvious methods for managing the tableau,
 * this class has methods to perform the Domino Robinson-Schensted procedure and its inverse.
 */
class Tableau {
        /**
         * The constructor creates a Tableau, usually from an Array of Dominos.<br>
         * Often you are constructing an empty tableau,
         * or making a Tableau from an array of Dominos (when cloning a tableau, for example).
         * There is also a version of the constructor which starts with a Tableau
         * and just gives you the same Tableau back again.
         * This is used by the derived class {@link TableauWithGrid}.
         * @param {Object} table
         * @param {Object} [table.tableau] - a Tableau to be copied.
         * @param {Object} [table.dominoList] - an Array of Dominos.
         */
        constructor(table) {
                table = table || {};
                if (table.tableau) {
                        /**
                         * an array of references to the Dominos of the tableau.
                         * @type {Domino[]}
                         */
                        this.dominoList = table.tableau.dominoList;
                        /**
                         * see {@link DominoGrid}
                         * @type {DominoGrid}
                         */
                        this.dominoGrid = table.tableau.dominoGrid;
                } else {
                        this.dominoList = table.dominoList || [];
                        this.dominoGrid = new DominoGrid(this.dominoList);
                }
        }

        /**
         *  makes a deep copy
         *  @return {Tableau}
         */
        clone() {
                return new Tableau({dominoList: Domino.cloneList(this.dominoList)});
        }

        /**
         * This creates an essentially empty Tableau of type B.
         * That is, the result has the zero Domino, and nothing else.
         * @return {Tableau}
         */
        static makeBTableau() {
                let zero = Domino.makeZero();
                return new Tableau({dominoList: [zero]});
        }

        /**
         *  This function uses the {@link TableauRendererDOM} class
         *  to draw the Tableau on a webpage.
         */
        draw() {
                document.body.appendChild(new TableauRendererDOM({tableau: this}).renderDOM());
        }

        /**
         * This function adds a Domino to the Tableau.
         * @param {Domino} domino - the Domino to add
         */
        insert(domino) {
                let inserted = false;
                for (let i = 0; i < this.dominoList.length; i++) {
                        if (this.dominoList[i].n > domino.n) {
                                this.dominoList.splice(i, 0, domino);
                                inserted = true;
                                break;
                        }
                }

                if (!inserted) {
                        this.dominoList.push(domino);
                }

                this.dominoGrid.addDomino(domino);
        }

        /**
         * This function adds a Domino to the Tableau. It has the same result as
         * the function {@link Tableau#insert}, but it uses a binary search to find the place
         * for the domino in {@link Tableau#dominoList}.
         * @param {Domino} domino - the Domino to add
         */
        insertBinary(domino) {
                let dominoList = this.dominoList;
                let listLength = dominoList.length;
                if (listLength == 0) {
                        dominoList.push(domino);
                        this.dominoGrid.addDomino(domino);
                        return;
                }

                let number = domino.n;
                let index;
                let last = listLength - 1;
                let lastNumber = dominoList[last].n;
                if (number > lastNumber) {
                        dominoList.push(domino);
                        this.dominoGrid.addDomino(domino);
                        return;
                }

                let first = 0;
                let mid = Math.floor((first + last)/ 2);
                while (true) {
                        let midNumber = dominoList[mid].n;
                        if (number < midNumber) {
                                if (mid == first) {
                                        index = mid;
                                        break;
                                }

                                let nextLowerNumber = dominoList[mid - 1].n;
                                if (number > nextLowerNumber) {
                                        index = mid;
                                        break;
                                }

                                // hence number < nextLowerNumber
                                if (first == mid - 1) {
                                        index = first;
                                        break;
                                }

                                last = mid - 1;
                                mid = Math.floor((first + last) / 2);
                        } else { // midNumber < number
                                if (last == mid + 1) {
                                        index = last;
                                        break;
                                }

                                first = mid + 1;
                                mid = Math.floor((first + last) / 2);
                        }
                }

                dominoList.splice(index, 0, domino);
                this.dominoGrid.addDomino(domino);
        }

        /**
         * This function adds a Domino to the Tableau, in a situation where the
         * domino can be added to the end of {@link Tableau#dominoList}.
         * That is, either the domino is being added to a Tableau which contains
         * numbers, and the number in the new domino is larger than any numbers
         * already present in the Tableau, or, alternatively, the tableau's
         * dominos do not contain numbers.
         * @param {Domino} domino - the Domino to add
         */
        insertAtEnd(domino) {
                this.dominoList.push(domino);
                this.dominoGrid.addDomino(domino);
        }

        /**
         * @param {number} i - the zero-based index of the row
         * @return {number} the length of the ith row
         */
        getRowLength(i) {
                return this.dominoGrid.getRowLength(i);
        }

        /**
         * @param {number} j - the zero-based index of the column
         * @return {number} the length of the jth column
         */
        getColumnLength(j) {
                return this.dominoGrid.getColumnLength(j);
        }

        /**
         * This function checks whether a domino shape
         * is at the edge of the Tableau.
         * @param {Object} position
         * @param {number} position.x
         * @param {number} position.y
         * @param {boolean} position.horizontal
         */
        isExtremal(position) {
                let x = position.x;
                let y = position.y;
                if (position.horizontal) {
                        let cond1 = this.getRowLength(y) == x + 2;
                        if (!cond1) return false;
                        let cond2 = this.getColumnLength(x) == y + 1;
                        if (!cond2) return false;
                        let cond3 = this.getColumnLength(x + 1) == y + 1;
                        if (!cond3) return false;
                        return true;
                } else {
                        let cond1 = this.getColumnLength(x) == y + 2;
                        if (!cond1) return false;
                        let cond2 = this.getRowLength(y) == x + 1;
                        if (!cond2) return false;
                        let cond3 = this.getRowLength(y + 1) == x + 1;
                        if (!cond3) return false;
                        return true;
                }
        }

        /**
         * This function adds one number with sign to the tableau,
         * using the Domino Robinson-Schensted procedure.
         * @param {number} rsNumber - the number can be positive or negative
         * @return {Object} {x: number, y: number, horizontal: boolean},<br>
         * an object holding the x and y coordinates
         * and the orientation of the change in the shape of the Tableau.
         */
        nextRobinsonSchensted(rsNumber) {
                let m = rsNumber > 0? rsNumber: -rsNumber;
                let newDomino = new Domino({n: m});
                let dominoGrid = this.dominoGrid;
                let grid = dominoGrid.grid;
                let position;
                let domino1;
                let domino2;

                /**
                 * [TODO getRowData description]
                 * @param  {number} y      [TODO description]
                 * @param  {number} number [TODO description]
                 * @return {Object}        [TODO description]
                 */
                function getRowData(y, number) {
                        for (let x = 0; ; x++) {
                                let domino1 = dominoGrid.get(x, y);
                                if (!domino1) {
                                        let rowPosition = {x:x, y:y, horizontal: true};
                                        return {position: rowPosition};
                                }

                                if (domino1.n > number) {
                                        let rowPosition = {x:x, y:y, horizontal: true};
                                        let domino2;
                                        if (domino1.horizontal) {
                                                domino2 = domino1;
                                        } else {
                                                domino2 = dominoGrid.get(x + 1, y);
                                        }

                                        return {position: rowPosition, domino1: domino1, domino2: domino2};
                                }
                        }
                }

                function getColumnData(x, number) {
                        for (let y = 0; ; y++) {
                                let domino1 = dominoGrid.get(x, y);
                                if (!domino1) {
                                        let columnPosition = {x:x, y:y, horizontal: false};
                                        return {position: columnPosition};
                                }

                                if (domino1.n > number) {
                                        let columnPosition = {x:x, y:y, horizontal: false};
                                        let domino2;
                                        if (!domino1.horizontal) {
                                                domino2 = domino1;
                                        } else {
                                                domino2 = dominoGrid.get(x, y + 1);
                                        }

                                        return {position: columnPosition, domino1: domino1, domino2: domino2};
                                }
                        }
                }

                let insertData;
                if (rsNumber > 0) {
                        insertData = getRowData(0, m);
                } else { // rsNumber < 0
                        insertData = getColumnData(0, m);
                }

                position = insertData.position;
                domino1 = insertData.domino1;
                domino2 = insertData.domino2;
                newDomino.x = position.x;
                newDomino.y = position.y;
                newDomino.horizontal = position.horizontal;
                this.insert(newDomino);

                while(domino1) {
                        if (position.horizontal) {
                                if (domino1.horizontal) {
                                        insertData = getRowData(domino1.y + 1, domino1.n);
                                        position = insertData.position;
                                        domino1.x = position.x;
                                        domino1.y = position.y;
                                        dominoGrid.addDomino(domino1);
                                        // dominoGrid.set(position.x, position.y, domino1);
                                        // dominoGrid.set(position.x + 1, position.y, domino1);
                                        domino1 = insertData.domino1;
                                        domino2 = insertData.domino2;
                                } else { // position.horizontal, !domino1.horizontal
                                        let nextDomino = dominoGrid.get(position.x + 1, position.y + 1);
                                        domino1.y += 1;
                                        domino1.horizontal = true;
                                        dominoGrid.set(position.x + 1, position.y + 1, domino1);
                                        position.x += 1;
                                        position.horizontal = false;
                                        domino1 = domino2;
                                        domino2 = nextDomino;
                                }
                        } else { // !position.horizontal
                                if (!domino1.horizontal) {
                                        insertData = getColumnData(domino1.x + 1, domino1.n);
                                        position = insertData.position;
                                        domino1.x = position.x;
                                        domino1.y = position.y;
                                        dominoGrid.addDomino(domino1);
                                        // dominoGrid.set(position.x, position.y, domino1);
                                        // dominoGrid.set(position.x, position.y + 1, domino1);
                                        domino1 = insertData.domino1;
                                        domino2 = insertData.domino2;
                                } else { // !position.horizontal, domino1.horizontal
                                        let nextDomino = dominoGrid.get(position.x + 1, position.y + 1);
                                        domino1.x += 1;
                                        domino1.horizontal = false;
                                        dominoGrid.set(position.x + 1, position.y + 1, domino1);
                                        position.y += 1;
                                        position.horizontal = true;
                                        domino1 = domino2;
                                        domino2 = nextDomino;
                                }
                        }
                }

                return position;
        }

        /**
         * This function reduces the tableau by the shape of one domino,
         * using the inverse Robinson-Schensted procedure.
         * @param {Object} inputPosition - an object holding the x and y coordinates
         * and the orientation of the change in the shape of the Tableau
         * @param {number} inputPosition.x
         * @param {number} inputPosition.y
         * @param {boolean} inputPosition.horizontal
         * @return {number} This is the number contained in the Domino
         * which is removed from the Tableau, if the removed domino is horizontal.
         * If the removed Domino is vertical, the number in the domino is multiplied by -1
         * before being returned.
         * @throws {Error} if the <code>inputPosition</code> is not valid for
         * this function
         */
        removeRobinsonSchensted(inputPosition) {
                if (!this.isExtremal(inputPosition)) {
                        throw new Error("Bad input to removeRobinsonSchensted");
                }

                let dominoGrid = this.dominoGrid;

                function getInsertColumn(row, number, startColumn) {
                        for (let testColumn = startColumn + 1; ; ++testColumn) {
                                let testNumber = dominoGrid.getContent(testColumn, row);
                                if (!testNumber || (testNumber > number)) {
                                        return testColumn - 1;
                                }
                        }
                }

                function getInsertRow(column, number, startRow) {
                        for (let testRow = startRow + 1; ; ++testRow) {
                                let testNumber = dominoGrid.getContent(column, testRow);
                                if (!testNumber || (testNumber > number)) {
                                        return testRow - 1;
                                }
                        }
                }

                let domino1;
                let domino2;
                if (inputPosition.horizontal) {
                        domino2 = dominoGrid.get(inputPosition.x + 1, inputPosition.y);
                        if (!domino2.horizontal) {
                                domino1 = dominoGrid.get(inputPosition.x, inputPosition.y);
                        }
                } else { // !inputPosition.horizontal
                        domino2 = dominoGrid.get(inputPosition.x, inputPosition.y + 1);
                        if (domino2.horizontal) {
                                domino1 = dominoGrid.get(inputPosition.x, inputPosition.y);
                        }
                }

                dominoGrid.removeExtremal(inputPosition);

                let position = {x: inputPosition.x, y: inputPosition.y, horizontal: inputPosition.horizontal};
                while (true) {
                        if (position.horizontal) {
                                if (position.y == 0) {
                                        break;
                                }

                                if (!domino2.horizontal) {
                                        let nextDomino = dominoGrid.get(position.x, position.y - 1);
                                        domino2.x -= 1;
                                        domino2.horizontal = true;
                                        dominoGrid.set(position.x, position.y - 1, domino2);
                                        position.y -= 1;
                                        position.horizontal = false;
                                        domino2 = domino1;
                                        domino1 = nextDomino;
                                } else { // domino2. horizontal
                                        let insertColumn = getInsertColumn(position.y - 1, domino2.n, position.x + 1);
                                        let movingDomino = domino2;
                                        position = {x: insertColumn - 1, y: position.y - 1, horizontal: true};
                                        domino2 = dominoGrid.get(position.x + 1, position.y);
                                        if (!domino2.horizontal) {
                                                domino1 = dominoGrid.get(position.x, position.y);
                                        }

                                        movingDomino.x = position.x;
                                        movingDomino.y = position.y;
                                        dominoGrid.addDomino(movingDomino);
                                        // dominoGrid.set(position.x, position.y, movingDomino);
                                        // dominoGrid.set(position.x + 1, position.y, movingDomino);
                                }
                        } else { // !position.horizontal
                                if (position.x == 0) {
                                        break;
                                }

                                if (domino2.horizontal) {
                                        let nextDomino = dominoGrid.get(position.x - 1, position.y);
                                        domino2.y -= 1;
                                        domino2.horizontal = false;
                                        dominoGrid.set(position.x - 1, position.y, domino2);
                                        position.x -= 1;
                                        position.horizontal = true;
                                        domino2 = domino1;
                                        domino1 = nextDomino;
                                } else { // domino2. horizontal
                                        let insertRow = getInsertRow(position.x - 1, domino2.n, position.y + 1);
                                        let movingDomino = domino2;
                                        position = {x: position.x - 1, y: insertRow - 1, horizontal:false};
                                        domino2 = dominoGrid.get(position.x, position.y + 1);
                                        if (domino2.horizontal) {
                                                domino1 = dominoGrid.get(position.x, position.y);
                                        }

                                        movingDomino.x = position.x;
                                        movingDomino.y = position.y;
                                        dominoGrid.addDomino(movingDomino);
                                        // dominoGrid.set(position.x, position.y, movingDomino);
                                        // dominoGrid.set(position.x, position.y + 1, movingDomino);
                                }
                        }
                }

                let index = this.dominoList.indexOf(domino2);
                this.dominoList.splice(index, 1);
                let number = domino2.n;
                if (!domino2.horizontal) {
                        number = -number;
                }

                return number;
        }

        /**
         * This function adds one number with sign to the tableau,
         * using the Domino Robinson-Schensted procedure.
         * It is the same as {@link Tableau#nextRobinsonSchensted},
         * except that it uses a binary search to find the position
         * to place a domino in a row or column, and it calls
         * {@link Tableau#insertBinary} to update {@link Tableau#dominoList}
         * with the new Domino.
         * @param {number} rsNumber - the number can be positive or negative
         * @return {Object} {x: number, y: number, horizontal: boolean},<br>
         * an object holding the x and y coordinates
         * and the orientation of the change in the shape of the Tableau.
         */
        nextRobinsonSchenstedBinary(rsNumber) {
                let m = rsNumber > 0? rsNumber: -rsNumber;
                let newDomino = new Domino({n: m});
                let dominoGrid = this.dominoGrid;
                let position;
                let domino1;
                let domino2;

                // binary search
                function getRowData(y, number, rowLength) {
                        if (rowLength == 0) {
                                let rowPosition = {x: 0, y: y, horizontal: true};
                                return {position: rowPosition};
                        }

                        let positionX;
                        let last = rowLength - 1;
                        let lastNumber = dominoGrid.getContent(last, y);
                        if (number > lastNumber) {
                                let rowPosition = {x: last + 1, y: y, horizontal: true};
                                return {position: rowPosition};
                        }

                        let first = 0;
                        let mid = Math.floor((first + last) / 2);
                        while (true) {
                                let midNumber = dominoGrid.getContent(mid, y);
                                let same = false;
                                if (number < midNumber) {
                                        if (mid == first) {
                                                positionX = mid;
                                                break;
                                        }

                                        let nextLowerNumber = dominoGrid.getContent(mid - 1, y);
                                        if (number == nextLowerNumber) {
                                                mid--;
                                                nextLowerNumber = dominoGrid.getContent(mid - 1, y);
                                                same = true;
                                        }

                                        if (number > nextLowerNumber) {
                                                positionX = mid;
                                                break;
                                        }

                                        // hence number < nextLowerNumber
                                        if (first == mid - 1) {

                                                positionX = first;
                                                break;
                                        }

                                        last = mid - 1;
                                        mid = Math.floor((first + last) / 2);
                                } else { // midNumber < number
                                        if (same) {
                                                mid++;
                                        } else {
                                                if (dominoGrid.getContent(mid + 1, y) == midNumber) {
                                                        mid++;
                                                }
                                        }

                                        if (last == mid + 1) {
                                                positionX = last;
                                                break;
                                        }

                                        first = mid + 1;
                                        mid = Math.floor((first + last) / 2);
                                }
                        }

                        let rowPosition = {x: positionX, y: y, horizontal: true};
                        let domino1 = dominoGrid.get(positionX, y);
                        let domino2;
                        if (domino1.horizontal) {
                                domino2 = domino1;
                        } else {
                                domino2 = dominoGrid.get(positionX + 1, y);
                        }

                        return {position: rowPosition, domino1: domino1, domino2: domino2};
                }

                // binary search
                function getColumnData(x, number, columnLength) {
                        if (columnLength == 0) {
                                let columnPosition = {x: x, y: 0, horizontal: false};
                                return {position: columnPosition};
                        }

                        let positionY;
                        let last = columnLength - 1;
                        let lastNumber = dominoGrid.getContent(x, last);
                        if (number > lastNumber) {
                                let columnPosition = {x: x, y: last + 1, horizontal: false};
                                return {position: columnPosition};
                        }

                        let first = 0;
                        let mid = Math.floor((first + last) / 2);
                        while (true) {
                                let midNumber = dominoGrid.getContent(x, mid);
                                let same = false;
                                if (number < midNumber) {
                                        if (mid == first) {
                                                positionY = mid;
                                                break;
                                        }

                                        let nextLowerNumber = dominoGrid.getContent(x, mid - 1);
                                        if (number == nextLowerNumber) {
                                                mid--;
                                                nextLowerNumber = dominoGrid.getContent(x, mid - 1);
                                                same = true;
                                        }

                                        if (number > nextLowerNumber) {
                                                positionY = mid;
                                                break;
                                        }

                                        // hence number < nextLowerNumber
                                        if (first == mid - 1) {
                                                positionY = first;
                                                break;
                                        }

                                        last = mid - 1;
                                        mid = Math.floor((first + last) / 2);
                                } else { // midNumber < number
                                        if (same) {
                                                mid++;
                                        } else {
                                                if (dominoGrid.getContent(x, mid + 1) == midNumber) {
                                                        mid++;
                                                }
                                        }

                                        if (last == mid + 1) {
                                                positionY = last;
                                                break;
                                        }

                                        first = mid + 1;
                                        mid = Math.floor((first + last) / 2);
                                }
                        }

                        let columnPosition = {x: x, y: positionY, horizontal: false};
                        let domino1 = dominoGrid.get(x, positionY);
                        let domino2;
                        if (!domino1.horizontal) {
                                domino2 = domino1;
                        } else {
                                domino2 = dominoGrid.get(x, positionY + 1);
                        }

                        return {position: columnPosition, domino1: domino1, domino2: domino2};
                }

                let insertData;
                if (rsNumber > 0) {
                        insertData = getRowData(0, m, this.getRowLength(0));
                } else { // rsNumber < 0
                        insertData = getColumnData(0, m, this.getColumnLength(0));
                }

                position = insertData.position;
                domino1 = insertData.domino1;
                domino2 = insertData.domino2;
                newDomino.x = position.x;
                newDomino.y = position.y;
                newDomino.horizontal = position.horizontal;
                this.insert(newDomino);

                while(domino1) {
                        if (position.horizontal) {
                                if (domino1.horizontal) {
                                        insertData = getRowData(domino1.y + 1, domino1.n, this.getRowLength(domino1.y + 1));
                                        position = insertData.position;
                                        domino1.x = position.x;
                                        domino1.y = position.y;
                                        dominoGrid.set(position.x, position.y, domino1);
                                        dominoGrid.set(position.x + 1, position.y, domino1);
                                        domino1 = insertData.domino1;
                                        domino2 = insertData.domino2;
                                } else { // position.horizontal, !domino1.horizontal
                                        let nextDomino = dominoGrid.get(position.x + 1, position.y + 1);
                                        domino1.y += 1;
                                        domino1.horizontal = true;
                                        dominoGrid.set(position.x + 1, position.y + 1, domino1);
                                        position.x += 1;
                                        position.horizontal = false;
                                        domino1 = domino2;
                                        domino2 = nextDomino;
                                }
                        } else { // !position.horizontal
                                if (!domino1.horizontal) {
                                        insertData = getColumnData(domino1.x + 1, domino1.n, this.getColumnLength(domino1.x + 1));
                                        position = insertData.position;
                                        domino1.x = position.x;
                                        domino1.y = position.y;
                                        dominoGrid.set(position.x, position.y, domino1);
                                        dominoGrid.set(position.x, position.y + 1, domino1);
                                        domino1 = insertData.domino1;
                                        domino2 = insertData.domino2;
                                } else { // !position.horizontal, domino1.horizontal
                                        let nextDomino = dominoGrid.get(position.x + 1, position.y + 1);
                                        domino1.x += 1;
                                        domino1.horizontal = false;
                                        dominoGrid.set(position.x + 1, position.y + 1, domino1);
                                        position.y += 1;
                                        position.horizontal = true;
                                        domino1 = domino2;
                                        domino2 = nextDomino;
                                }
                        }
                }

                return position;
        }

        /**
         * This function reduces the tableau by the shape of one domino,
         * using the inverse Robinson-Schensted procedure.
         * It is the same as {@link Tableau#removeRobinsonSchensted},
         * except that it uses a binary search
         * to find the position to place a domino in a row or column.
         * Also, it uses a binary search instead of <code>indexOf</code>
         * to remove a Domino from {@link Tableau#dominoList}.
         * @param {Object} inputPosition - an object holding the x and y coordinates
         * and the orientation of the change in the shape of the Tableau
         * @param {number} inputPosition.x
         * @param {number} inputPosition.y
         * @param {boolean} inputPosition.horizontal
         * @return {number} This is the number contained in the Domino
         * which is removed from the Tableau, if the removed domino is horizontal.
         * If the removed Domino is vertical, the number in the domino is multiplied by -1
         * before being returned.
         * @throws {Error} if the <code>inputPosition</code> is not valid for
         * this function
         */
        removeRobinsonSchenstedBinary(inputPosition) {
                if (!this.isExtremal(inputPosition)) {
                        throw new Error("Bad input to removeRobinsonSchensted");
                }

                let dominoGrid = this.dominoGrid;

                function getInsertColumn(row, number, startColumn, rowLength) {
                        let positionX;
                        let last = rowLength - 1;
                        let lastNumber = dominoGrid.getContent(last, row);
                        if (number > lastNumber) {
                                return last;
                        }

                        // invariant: firstNumber < number < lastNumber
                        let first = startColumn;
                        let mid = Math.floor((first + last) / 2);
                        while (true) {
                                let midNumber = dominoGrid.getContent(mid, row);
                                if (number > midNumber) {
                                        if (mid + 1 == last) {
                                                positionX = mid;
                                                break;
                                        }

                                        let nextHigherNumber = dominoGrid.getContent(mid + 1, row);
                                        if (nextHigherNumber == midNumber) {
                                                mid++;
                                                nextHigherNumber = dominoGrid.getContent(mid + 1, row);
                                        }

                                        if (number < nextHigherNumber) {
                                                positionX = mid;
                                                break;
                                        }

                                        first = mid + 1;
                                        mid = Math.floor((first + last) / 2);
                                } else { // midNumber > number
                                        if (first == mid - 1) {
                                                positionX = first;
                                                break;
                                        }

                                        last = mid;
                                        mid = Math.floor((first + last) / 2);
                                        if (first == mid) {
                                                positionX = mid;
                                                break;
                                        }
                                }
                        }

                        return positionX;
                }

                function getInsertRow(column, number, startRow, columnLength) {
                        let positionY;
                        let last = columnLength - 1;
                        let lastNumber = dominoGrid.getContent(column, last);
                        if (number > lastNumber) {
                                return last;
                        }

                        // invariant: firstNumber < number < lastNumber
                        let first = startRow;
                        let mid = Math.floor((first + last) / 2);
                        while (true) {
                                let midNumber = dominoGrid.getContent(column, mid);
                                if (number > midNumber) {
                                        if (mid + 1 == last) {
                                                positionY = mid;
                                                break;
                                        }

                                        let nextHigherNumber = dominoGrid.getContent(column, mid + 1);
                                        if (nextHigherNumber == midNumber) {
                                                mid++;
                                                nextHigherNumber = dominoGrid.getContent(column, mid + 1);
                                        }

                                        if (number < nextHigherNumber) {
                                                positionY = mid;
                                                break;
                                        }

                                        first = mid + 1;
                                        mid = Math.floor((first + last) / 2);
                                } else { // midNumber > number
                                        if (first == mid - 1) {
                                                positionY = first;
                                                break;
                                        }

                                        last = mid;
                                        mid = Math.floor((first + last) / 2);
                                        if (first == mid) {
                                                positionY = mid;
                                                break;
                                        }
                                }
                        }

                        return positionY;
                }

                let domino1;
                let domino2;
                if (inputPosition.horizontal) {
                        domino2 = dominoGrid.get(inputPosition.x + 1, inputPosition.y);
                        if (!domino2.horizontal) {
                                domino1 = dominoGrid.get(inputPosition.x, inputPosition.y);
                        }
                } else { // !inputPosition.horizontal
                        domino2 = dominoGrid.get(inputPosition.x, inputPosition.y + 1);
                        if (domino2.horizontal) {
                                domino1 = dominoGrid.get(inputPosition.x, inputPosition.y);
                        }
                }

                dominoGrid.removeExtremal(inputPosition);

                let position = {x: inputPosition.x, y: inputPosition.y, horizontal: inputPosition.horizontal};
                while (true) {
                        if (position.horizontal) {
                                if (position.y == 0) {
                                        break;
                                }

                                if (!domino2.horizontal) {
                                        let nextDomino = dominoGrid.get(position.x, position.y - 1);
                                        domino2.x -= 1;
                                        domino2.horizontal = true;
                                        dominoGrid.set(position.x, position.y - 1, domino2);
                                        position.y -= 1;
                                        position.horizontal = false;
                                        domino2 = domino1;
                                        domino1 = nextDomino;
                                } else { // domino2. horizontal
                                        let insertColumn = getInsertColumn(position.y - 1, domino2.n, position.x + 1, this.getRowLength(position.y - 1));
                                        let movingDomino = domino2;
                                        position = {x: insertColumn - 1, y: position.y - 1, horizontal: true};
                                        domino2 = dominoGrid.get(position.x + 1, position.y);
                                        if (!domino2.horizontal) {
                                                domino1 = dominoGrid.get(position.x, position.y);
                                        }

                                        movingDomino.x = position.x;
                                        movingDomino.y = position.y;
                                        dominoGrid.set(position.x, position.y, movingDomino);
                                        dominoGrid.set(position.x + 1, position.y, movingDomino);
                                }
                        } else { // !position.horizontal
                                if (position.x == 0) {
                                        break;
                                }

                                if (domino2.horizontal) {
                                        let nextDomino = dominoGrid.get(position.x - 1, position.y);
                                        domino2.y -= 1;
                                        domino2.horizontal = false;
                                        dominoGrid.set(position.x - 1, position.y, domino2);
                                        position.x -= 1;
                                        position.horizontal = true;
                                        domino2 = domino1;
                                        domino1 = nextDomino;
                                } else { // domino2. horizontal
                                        let insertRow = getInsertRow(position.x - 1, domino2.n, position.y + 1, this.getColumnLength(position.x - 1));
                                        let movingDomino = domino2;
                                        position = {x: position.x - 1, y: insertRow - 1, horizontal:false};
                                        domino2 = dominoGrid.get(position.x, position.y + 1);
                                        if (domino2.horizontal) {
                                                domino1 = dominoGrid.get(position.x, position.y);
                                        }

                                        movingDomino.x = position.x;
                                        movingDomino.y = position.y;
                                        dominoGrid.set(position.x, position.y, movingDomino);
                                        dominoGrid.set(position.x, position.y + 1, movingDomino);
                                }
                        }
                }

                // let index = this.dominoList.indexOf(domino2);
                let dominoList = this.dominoList;
                let number = domino2.n;
                let index;
                let last = dominoList.length - 1;
                let first = 0;
                let mid = Math.floor((first + last)/ 2);
                while (true) {
                        let midNumber = dominoList[mid].n;
                        if (number == midNumber) {
                                index = mid;
                                break;
                        }

                        if (number < midNumber) {
                                last = mid - 1;
                        } else { // midNumber < number
                                first = mid + 1;
                        }

                        mid = Math.floor((first + last) / 2);
                }


                this.dominoList.splice(index, 1);
                // let number = domino2.n;
                if (!domino2.horizontal) {
                        number = -number;
                }

                return number;
        }
}

/**
 * For most application of domino tableaux, we need to associate
 * a type (B, C, or D) to the tableau.  Pictorially, this places the tableau
 * on a grid of 2x2 squares.  In particular, this allows us to move the tableau
 * through cycles.
 * @extends {Tableau}
 */
class TableauWithGrid extends Tableau {
        /**
         * @param {Object} table
         * @param {string} table.type - One of "B", "C", or "D"
         * @param {Object} [table.tableau] - a Tableau to be copied.
         * @param {Object} [table.dominoList] - an Array of Dominos.
         */
         constructor(table) {
                 if (table.type == "B" && !(table.tableau || table.dominoList)) {
                         super({dominoList: [Domino.makeZero()]});
                 } else {
                         super(table);
                 }
                 /**
                  * One of "B", "C", or "D"
                  * @type {string}
                  */
                 this.type = table.type;
                 /**
                  * See {@link TableauWithGrid.getGrid} for a description of this function.
                  * The type of the TableauWithGrid is mostly expressed through
                  * this function, which is used to determine where a square is
                  * situated in the 2x2 box of the grid which contains it.
                  * @type {function}
                  */
                 this.getGridSubPosition = TableauWithGrid.getGrid(this.type);
         }

        /**
         *  makes a deep copy
         *  @return {TableauWithGrid}
         */
        clone() {
                return new TableauWithGrid({type: this.type, dominoList: Domino.cloneList(this.dominoList)});
        }

        /**
         * Grids are here implemented as functions which specify how a given square
         * is located in relation to a grid of 2x2 boxes.
         * These function accepts an object with x and y members,
         * that is, the location of the square,
         * and return a string, one of "X", "Y", "Z", or "W".
         * This return value indicates the location of the square within the
         * 2x2 box of the grid which contains it.
         * "X" indicates the upper-left corner of the 2x2 box,
         * "Y" the upper-right, "Z" the lower-left,
         * and "W" the lower-right.
         * @param {string} type - one of "B", "C", or "D"
         * @return {function}
         */
        static getGrid(type) {
                const GRIDS = {
                        B: function getGridSubPosition(square) {
                                if (square.x % 2 == 1 && square.y % 2 == 1) return "X";
                                if (square.x % 2 == 0 && square.y % 2 == 1) return "Y";
                                if (square.x % 2 == 1 && square.y % 2 == 0) return "Z";
                                if (square.x % 2 == 0 && square.y % 2 == 0) return "W";
                        },
                        C: function getGridSubPosition(square) {
                                if (square.x % 2 == 0 && square.y % 2 == 0) return "X";
                                if (square.x % 2 == 1 && square.y % 2 == 0) return "Y";
                                if (square.x % 2 == 0 && square.y % 2 == 1) return "Z";
                                if (square.x % 2 == 1 && square.y % 2 == 1) return "W";
                        },
                        D: function getGridSubPosition(square) {
                                if (square.x % 2 == 1 && square.y % 2 == 0) return "X";
                                if (square.x % 2 == 0 && square.y % 2 == 0) return "Y";
                                if (square.x % 2 == 1 && square.y % 2 == 1) return "Z";
                                if (square.x % 2 == 0 && square.y % 2 == 1) return "W";
                        }
                }

                return GRIDS[type];
        }

        /**
         * These offsets are used by the drawing classes
         * such as {@link TableauRendererDOM}
         * to situate a TableauWithGrid in its grid.
         * @param {string} type - one of "B", "C", or "D"
         * @return {Object} {x: number, y: number}<br>
         * These are the x and y coordinates of the top-left corner
         * of the tableau in relation to the 2x2 grid.
         */
        static getOffset(type) {
                const OFFSETS = {
                        B: {x: 1, y: 1},
                        C: {x: 0, y: 0},
                        D: {x: 1, y: 0}
                }

                return OFFSETS[type];
        }

        /**
         * @param {Domino} domino
         * @return {Square} the x and y coordinates of the domino's
         * variable square with respect to the grid.
         */
        getVariableSquare(domino) {
                let gridSubPosition = this.getGridSubPosition(domino);
                if (gridSubPosition == "X" || gridSubPosition == "W") {
                        return new Square({x: domino.x, y: domino.y});
                }

                if (domino.horizontal) {
                        return new Square({x: domino.x + 1, y: domino.y});
                }

                return new Square({x: domino.x, y: domino.y + 1});
        }

        /**
         * @param {Domino} domino
         * @return {Square} the x and y coordinates of the domino's
         * fixed square with respect to the grid.
         */
        getFixedSquare(domino) {
                let gridSubPosition = this.getGridSubPosition(domino);
                if (gridSubPosition == "Y" || gridSubPosition == "Z") {
                        return new Square({x: domino.x, y: domino.y});
                }

                if (domino.horizontal) {
                        return new Square({x: domino.x + 1, y: domino.y});
                }
                return new Square({x: domino.x, y: domino.y + 1});
        }

        /**
         * @param {Domino} domino
         * @return {boolean} If true, the domino is boxed with respect to the grid.
         */
        isBoxed(domino) {
                let gridSubPosition = this.getGridSubPosition(domino);
                if (gridSubPosition == "X") {
                        return true;
                }

                if (domino.horizontal && gridSubPosition == "Z") {
                        return true;
                }

                if (!domino.horizontal && gridSubPosition == "Y") {
                        return true;
                }

                return false;
        }

        /**
         * This function, given the back square of an open cycle in the
         * tableau with grid, returns the forward square of that cycle.
         * @param {Square} backSquare - the back square of the cycle
         * @return {Square} the forward square of the cycle
         */
        getForwardSquare(backSquare) {
                let grid = this.dominoGrid;
                let square = backSquare.clone(); //{x: backSquare.x, y:backSquare.y};
                let boxed = this.getGridSubPosition(square) == "W";
                let domino = grid.get(square.x, square.y);
                do {
                        let fixedSquare = this.getFixedSquare(domino);
                        let x = fixedSquare.x;
                        let y = fixedSquare.y;
                        let gridSubPosition = this.getGridSubPosition(fixedSquare);
                        if ( (boxed && gridSubPosition == "Y") || (!boxed && gridSubPosition == "Z") ) {
                                let comp = y == 0 ? null : grid.get(x + 1, y - 1);
                                if (y == 0 || (comp && (comp.n < domino.n))) {
                                        square.x = x + 1;
                                        square.y = y;
                                        domino = grid.get(square.x, square.y);
                                        if (!domino) break;
                                } else {
                                        square.x = x;
                                        square.y = y - 1;
                                        domino = grid.get(square.x, square.y);
                                }
                        } else {
                                let comp = x == 0 ? null : grid.get(x - 1, y + 1);
                                if (x == 0 || (comp && (comp.n < domino.n))) {
                                        square.x = x;
                                        square.y = y + 1;
                                        domino = grid.get(square.x, square.y);
                                        if (!domino) break;
                                } else {
                                        square.x = x - 1;
                                        square.y = y;
                                        domino = grid.get(square.x, square.y);
                                }
                        }
                } while (true);

                return square;
        }

        /**
         * This function, given the forward square of an open cycle in the
         * tableau with grid, returns the back square of that cycle.
         * @param {Square} forwardSquare - the forward square of the cycle
         * @return {Square} the back square of the cycle
         */
        getBackSquare(forwardSquare) {
                let grid = this.dominoGrid;
                let square = forwardSquare.clone();//{x: forwardSquare.x, y:forwardSquare.y};
                let boxed = this.getGridSubPosition(square) == "X";
                let domino;
                do {
                        let x = square.x;
                        let y = square.y;
                        let gridSubPosition = this.getGridSubPosition(square);
                        if ((boxed && gridSubPosition == "X") || (!boxed && gridSubPosition == "W" ) ) {
                                let choice1 = y == 0 ? null : grid.get(x, y - 1);
                                let choice2 = x == 0 ? null : grid.get(x - 1, y);
                                if (!choice1 && !choice2) {
                                        // case x ==0 && y == 0
                                } else if (!choice2 || (choice1 && (choice1.n > choice2.n))) {  //for later, case x or y equals 0
                                        domino = choice1;
                                        square = this.getVariableSquare(domino);
                                } else {
                                        domino = choice2;
                                        square = this.getVariableSquare(domino);
                                }
                        } else {
                                let choice1 = grid.get(x, y + 1);
                                let choice2 = grid.get(x + 1, y);
                                if (!choice1 && !choice2) {
                                        break;
                                } else if (!choice2 || (choice1 && (choice1.n < choice2.n))) {
                                        domino = choice1;
                                        square = this.getVariableSquare(domino);
                                } else {
                                        domino = choice2;
                                        square = this.getVariableSquare(domino);
                                }
                        }
                } while (true);

                return square;
        }

        /**
         * This function moves the tableau with grid through an open cycle
         * specified by its back square.
         * @param {Square} backSquare -the back square of the cycle
         * @return {Object} {forwardSquare: {@link Square}, movedDominos: {@link Array.<Domino>}}<br>
         * forwardSquare - the forward square of the cycle <br>
         * movedDominos - an array of the Dominos in the cycle
         */
        moveOpenCycle(backSquare) {
                let movedDominos = [];
                let dominoGrid = this.dominoGrid;
                let square = backSquare.clone();//{x: backSquare.x, y:backSquare.y};
                let boxed = this.getGridSubPosition(square) == "W";
                let bZero = false;
                let cZero = false;
                if (square.x == 0 && square.y ==0 ) {
                        boxed = true;
                        cZero = true;
                }

                let domino = dominoGrid.get(square.x, square.y);
                if (!cZero) {
                        dominoGrid.unset(square.x, square.y);
                }

                do {
                        let fixedSquare = this.getFixedSquare(domino);
                        let x = fixedSquare.x;
                        let y = fixedSquare.y;
                        let gridSubPosition = this.getGridSubPosition(fixedSquare);
                        if ( (boxed && gridSubPosition == "Y") || (!boxed && gridSubPosition == "Z") ) {
                                let comp = y == 0 ? null : dominoGrid.get(x + 1, y - 1);
                                if (y == 0 || (comp && (comp.n < domino.n))) {
                                        square.x = x + 1;
                                        square.y = y;
                                        let nextDomino = dominoGrid.get(square.x, square.y);
                                        if (domino.horizontal) {
                                                this.slideDominoForCycle(domino);
                                        } else {
                                                this.flipDominoForCycle(domino);
                                        }

                                        movedDominos.push(domino);
                                        if (!nextDomino) {
                                                break;
                                        }

                                        domino = nextDomino;
                                } else {
                                        square.x = x;
                                        square.y = y - 1;
                                        let nextDomino = dominoGrid.get(square.x, square.y);
                                        if (!domino.horizontal) {
                                                this.slideDominoForCycle(domino);
                                        } else {
                                                this.flipDominoForCycle(domino);
                                        }

                                        movedDominos.push(domino);
                                        domino = nextDomino;
                                        if (domino.zero) {
                                                movedDominos.push(domino);
                                                bZero = true;
                                                break;
                                        }
                                }
                        } else {
                                let comp = x == 0 ? null : dominoGrid.get(x - 1, y + 1);
                                if (x == 0 || (comp && (comp.n < domino.n))) {
                                        square.x = x;
                                        square.y = y + 1;
                                        let nextDomino = dominoGrid.get(square.x, square.y);
                                        if (!domino.horizontal) {
                                                this.slideDominoForCycle(domino);
                                        } else {
                                                this.flipDominoForCycle(domino);
                                        }

                                        movedDominos.push(domino);
                                        if (!nextDomino) {
                                                break;
                                        }

                                        domino = nextDomino;
                                } else {
                                        square.x = x - 1;
                                        square.y = y;
                                        let nextDomino = dominoGrid.get(square.x, square.y);
                                        if (domino.horizontal) {
                                                this.slideDominoForCycle(domino);
                                        } else {
                                                this.flipDominoForCycle(domino);
                                        }

                                        movedDominos.push(domino);
                                        domino = nextDomino;
                                        if (domino.zero) {
                                                movedDominos.push(domino);
                                                bZero = true;
                                                break;
                                        }
                                }
                        }
                } while (true);

                if (bZero) {
                        this.dominoList.shift();
                        this.type = "C";
                        this.getGridSubPosition = TableauWithGrid.getGrid("C");
                }

                if (cZero) {
                        this.insert(Domino.makeZero());
                        this.type = "B";
                        this.getGridSubPosition = TableauWithGrid.getGrid("B");
                }

                let forwardSquare = square;
                return {forwardSquare, movedDominos};
        }

        /**
         * This function moves the tableau with grid through an open cycle
         * specified by its forward square.
         * @param {Square} forwardSquare - the forward square of the cycle
         * @return {Object} {backSquare: {@link Square}, movedDominos: {@link Array.<Domino>}}<br>
         * backSquare - the back square of the cycle <br>
         * movedDominos - an array of the Dominos in the cycle
         */
        moveOpenCycleToSquare(forwardSquare) {
                let movedDominos = [];
                let dominoGrid = this.dominoGrid;
                let square = {x: forwardSquare.x, y:forwardSquare.y};
                let boxed = this.getGridSubPosition(square) == "X";
                let bZero = false;
                let cZero = false;
                //type B
                if (square.x == 0 && square.y == 0) {
                        boxed = true;
                        bZero = true;
                        movedDominos.push(dominoGrid.get(0, 0));
                }

                let domino;
                do {
                        let x = square.x;
                        let y = square.y;
                        let gridSubPosition = this.getGridSubPosition(square);
                        if ((boxed && gridSubPosition == "X") || (!boxed && gridSubPosition == "W" ) ) {
                                let choice1 = y == 0 ? null : dominoGrid.get(x, y - 1);
                                let choice2 = x == 0 ? null : dominoGrid.get(x - 1, y);
                                if (!choice1 && !choice2) {
                                        // case x == 0 && y == 0
                                        cZero = true;
                                        break;
                                } else if (!choice2 || (choice1 && (choice1.n > choice2.n))) {  //for later, case x or y equals 0
                                        domino = choice1;
                                        square = this.getVariableSquare(domino);
                                        if (!domino.horizontal) {
                                                this.slideDominoForCycle(domino);
                                        } else {
                                                this.flipDominoForCycle(domino);
                                        }

                                        movedDominos.push(domino);
                                } else {
                                        domino = choice2;
                                        square = this.getVariableSquare(domino);
                                        if (domino.horizontal) {
                                                this.slideDominoForCycle(domino);
                                        } else {
                                                this.flipDominoForCycle(domino);
                                        }

                                        movedDominos.push(domino);
                                }
                        } else {
                                let choice1 = dominoGrid.get(x, y + 1);
                                let choice2 = dominoGrid.get(x + 1, y);
                                if (!choice1 && !choice2) {
                                        break;
                                } else if (!choice2 || (choice1 && (choice1.n < choice2.n))) {
                                        domino = choice1;
                                        square = this.getVariableSquare(domino);
                                        if (!domino.horizontal) {
                                                this.slideDominoForCycle(domino);
                                        } else {
                                                this.flipDominoForCycle(domino);
                                        }

                                        movedDominos.push(domino);
                                } else {
                                        domino = choice2;
                                        square = this.getVariableSquare(domino);
                                        if (domino.horizontal) {
                                                this.slideDominoForCycle(domino);
                                        } else {
                                                this.flipDominoForCycle(domino);
                                        }

                                        movedDominos.push(domino);
                                }
                        }
                } while (true);

                if (!cZero) {
                        dominoGrid.unset(square.x, square.y);
                }

                if (bZero) {
                        this.dominoList.shift();
                        this.type = "C";
                        this.getGridSubPosition = TableauWithGrid.getGrid("C");
                }

                if (cZero) {
                        this.insert(Domino.makeZero());
                        this.type = "B";
                        this.getGridSubPosition = TableauWithGrid.getGrid("B");
                }

                let backSquare = square;
                return {backSquare, movedDominos};
        }

        /**
         * helper function for {@link TableauWithGrid#moveOpenCycle}
         * and {@link TableauWithGrid#moveOpenCycleToSquare}
         */
        flipDominoForCycle(domino) {
                let dominoGrid = this.dominoGrid;
                let x = domino.x;
                let y = domino.y;
                let fixedSquare = this.getFixedSquare(domino);
                if (domino.horizontal) {
                        domino.horizontal = false;
                        if (fixedSquare.x == x) {
                                dominoGrid.set(x, y + 1, domino);
                        } else { // fixedSquare.x != x
                                domino.x = x + 1;
                                domino.y = y - 1;
                                dominoGrid.set(x + 1, y - 1, domino);
                        }
                } else { //!domino.horizontal
                        domino.horizontal = true;
                        if (fixedSquare.y == y) {
                                dominoGrid.set(x + 1, y, domino);
                        } else { // fixedSquare.x != x
                                domino.x = x - 1;
                                domino.y = y + 1;
                                dominoGrid.set(x - 1, y + 1, domino);
                        }
                }
        }

        /**
        * helper function for {@link TableauWithGrid#moveOpenCycle}
        * and {@link TableauWithGrid#moveOpenCycleToSquare}
         */
        slideDominoForCycle(domino) {
                let dominoGrid = this.dominoGrid;
                let x = domino.x;
                let y = domino.y;
                let fixedSquare = this.getFixedSquare(domino);
                if (domino.horizontal) {
                        if (fixedSquare.x == x) {
                                domino.x = x - 1;
                                dominoGrid.set(x - 1, y, domino);
                        } else { // fixedSquare.x != x
                                domino.x = x + 1;
                                dominoGrid.set(x + 2, y, domino);
                        }
                } else { //!domino.horizontal
                        if (fixedSquare.y == y) {
                                domino.y = y - 1;
                                dominoGrid.set(x, y - 1, domino);
                        } else { // fixedSquare.x != x
                                domino.y = y + 1;
                                dominoGrid.set(x, y + 2, domino);
                        }
                }
        }
}

/**
 * This class handles the input to the Domino Robinson-Schensted algorithm,
 * which is a signed permutation.  A signed permutation can also be represented
 * as a string, and is for display purposes.
 * So, this class contains functions to go between the two representations
 * of the permutation.
 */
class ParameterDominoRS {
        /**
         * This constructs a {@link ParameterDominoRS} from either
         * an array of numbers or a space-separated string of those numbers.
         * If input is not given, the constructor returns the representation
         * of the empty signed permutation.
         * {@link TableauSignsSppq}.
         * @param {Object} table
         * @param {number[]} [table.array] - an array of the type required
         * for {@link ParameterDominoRS#parameter}
         * @param {string} [table.parameterString] - the string representation
         * of a signed permutation.
         */
        constructor(table) {
                table = table || {};
                if (table.array) {
                        /**
                         * This is an array which stores the signed permutation.
                         * Functionally, the array is 1-based, that is,
                         * parameter[0] = 0, basically just to discard that entry,
                         * and parameter[i] = j means that the permutation takes i to j.
                         * @type {Array.number}
                         */
                        this.parameter = table.array;
                } else if (table.parameterString) {
                        this.parameter = ParameterDominoRS.parse(table.parameterString).parameter;
                } else {
                        this.parameter = [0];
                }
        }

        /**
         *  makes a deep copy
         *  @return {ParameterDominoRS}
         */
        clone() {
                let parameter = [];
                this.parameter.forEach((item) => parameter.push(item));
                return new ParameterDominoRS({array: parameter});
        }

        /**
         * This function generates a random signed permutation
         * and stores it in a ParameterDominoRS object.
         * @param {number} n - the size of the signed permutation
         * @return {ParameterDominoRS}
         */
        static generateParameter(n) {
                let chooseOne = function(array) {
                        let choiceIndex = Math.floor(Math.random() * array.length);
                        return array.splice(choiceIndex, 1)[0];
                }

                let numbers = [];
                for (let i = 1; i <= n; i++) { numbers.push(i); }

                let parameter = [];
                parameter[0] = 0;

                for (let index = 1; index <= n; index++) {
                        let number = chooseOne(numbers);
                        if (Math.random() < .5) {
                                number *= -1;
                        }

                        parameter[index] = number;
                }

                return new ParameterDominoRS({array: parameter});
        }

        /**
         * This function gets the string representations
         * of the signed permutation.
         * @return {string} the string representation
         */
        getParameterString() {
                let parameter = this.parameter;
                let output = "";
                for (let index = 1; index < parameter.length; index++) {
                         output += parameter[index] + " ";
                }

                output = output.slice(0, -1);
                return output;
        }

        /**
         * This function takes the string representation
         * of a signed permutation and returns the {@link ParameterDominoRS} object
         * associated to that signed permutation.
         * @param {string} parameterString - a string representing a signed permutation
         * @return {ParameterDominoRS} a {@link ParameterDominoRS} object
         * representing the same signed permutation
         */
        static parse(parameterString) {
                let data = parameterString.split(" ").filter(x => x);
                let parameter = [0];
                data.forEach((datum) => {
                        parameter.push(parseInt(datum));
                });
                return new ParameterDominoRS({array: parameter});
        }
}

/**
 * Class to store a pair of {@link Tableau}.
*/
class TableauPair {
        /**
         * @param {Object} table
         * @param {Tableau} [table.left] - the left tableau of the pair.
         * @param {Tableau} [table.right] - the right tableau of the pair.
         */
        constructor(table) {
                table = table || {};
                if (!table.left) {
                        /**
                         * the left tableau of the pair
                         * @type {Tableau}
                         */
                        this.left = new Tableau();
                        /**
                         * the right tableau of the pair
                         * @type {Tableau}
                         */
                        this.right = new Tableau();
                } else {
                        this.left = table.left;
                        this.right = table.right;
                }
        }

        /**
         * This function makes an empty {@link TableauPair} of the B type, that is,
         * with zero squares in the top-left corners of the tableaux.
         * @return {TableauPair}
         */
        static makeTableauPairB() {
                let left = Tableau.makeBTableau();
                let right = Tableau.makeBTableau();
                return new TableauPair({left, right});
        }

        /**
         * makes a deep copy
         * @return {TableauPair}
         */
        clone() {
                let left = this.left.clone();
                let right = this.right.clone();
                return new TableauPair({left, right});
        }

        /**
         *  This function uses the {@link TableauPairRendererDOM} class
         *  to draw the {@link TableauPair} on a webpage.
         */
        draw() {
                document.body.appendChild(new TableauPairRendererDOM({tableauPair: this}).renderDOM());
        }

        /**
         * This function adds one number to the left tableau,
         * using {@link Tableau#nextRobinsonSchensted},
         * that is, using the Domino Robinson-Schensted procedure.
         * It then adds a domino to the right tableau to record
         * the shape change caused by adding the number to the left tableau.
         * @param {number} numberLeft - can be positive or negative
         * @param {number} numberRight - a positive integer
         */
        nextRobinsonSchensted(numberLeft, numberRight) {
                let position = this.left.nextRobinsonSchensted(numberLeft);
                let dominoRight = new Domino(position);
                dominoRight.n = numberRight;
                this.right.insertAtEnd(dominoRight);
        }

        /**
         * This static function performs the entire Domino Robinson-Schensted algorithm.
         * That is, it takes as input a signed permutation and outputs a pair of tableaux.
         * @param {ParameterDominoRS} parameterObject - the signed permutation
         * wrapped in a ParameterDominoRS object.
         * @return {TableauPair}
         */
        static RobinsonSchensted(parameterObject) {
                let parameter = parameterObject.parameter;
                let myTableauPair = new TableauPair();
                for (let index = 1; index < parameter.length; ++index) {
                        myTableauPair.nextRobinsonSchensted(parameter[index], index);
                }

                return myTableauPair;
        }

        /**
         * This static function performs the Domino Robinson-Schensted algorithm,
         * just as the function {@link TableauPair.RobinsonSchensted} does.
         * In addition, it draws each step.
         * @param {ParameterDominoRS} parameterObject - the signed permutation
         * wrapped in a ParameterDominoRS object.
         * @return {TableauPair}
         */
        static RobinsonSchenstedDrawSteps(parameterObject) {
                let parameter = parameterObject.parameter;
                let myTableauPair = new TableauPair();
                for (let index = 1, length = parameter.length; index < length; ++index) {
                        Page.displayText(parameter[index]);
                        myTableauPair.nextRobinsonSchensted(parameter[index], index);
                        myTableauPair.draw();
                }

                return myTableauPair;
        }

        /**
         * This static function performs the entire Domino Robinson-Schensted algorithm.
         * That is, it takes as input a signed permutation and outputs a pair of tableaux.
         * In this case, however, the output is a pair of {@link TableauWithGrid}.
         * @param {ParameterDominoRS} parameterObject - the signed permutation
         * wrapped in a ParameterDominoRS object
         * @param {string} type - "B", "C", or "D"
         * @return {TableauPair}
         */
        static RobinsonSchenstedGrid(parameterObject, type) {
                let parameter = parameterObject.parameter;
                let myTableauPair;
                if (type == "B") {
                        myTableauPair = TableauPair.makeTableauPairB();
                } else {
                        myTableauPair = new TableauPair();
                }

                myTableauPair.left = new TableauWithGrid({type, tableau: myTableauPair.left});
                myTableauPair.right = new TableauWithGrid({type, tableau: myTableauPair.right});
                for (let index = 1, length = parameter.length; index < length; ++index) {
                        myTableauPair.nextRobinsonSchensted(parameter[index], index);
                }

                return myTableauPair;
        }

        /**
         * This static function performs the Domino Robinson-Schensted algorithm,
         * just as the function {@link TableauPair.RobinsonSchenstedGrid} does.
         * So, again, the output is a pair of {@link TableauWithGrid}.
         * In addition, it draws each step.  So, the tableaux are drawn with their grid.
         * @param {ParameterDominoRS} parameterObject - the signed permutation
         * wrapped in a ParameterDominoRS object
         * @return {TableauPair}
         */
        static RobinsonSchenstedGridDrawSteps(parameterObject, type) {
                let parameter = parameterObject.parameter;
                let myTableauPair;
                if (type == "B") {
                        myTableauPair = TableauPair.makeTableauPairB();
                } else {
                        myTableauPair = new TableauPair();
                }

                myTableauPair.left = new TableauWithGrid({type, tableau: myTableauPair.left});
                myTableauPair.right = new TableauWithGrid({type, tableau: myTableauPair.right});
                for (let index = 1, length = parameter.length; index < length; ++index) {
                        Page.displayText(parameter[index]);
                        myTableauPair.nextRobinsonSchensted(parameter[index], index);
                        myTableauPair.draw();
                }

                return myTableauPair;
        }

        /**
         * This function performs the inverse of the Domino Robinson-Schensted algorithm.
         * That is, it starts with a pair of tableaux and returns a signed permutation.
         * @return {ParameterDominoRS} the signed permutation wrapped in a
         * ParameterDominoRS object.
         */
        getParameter() {
                let size = this.right.dominoList.length;
                let end = 0;
                let skip = 1;
                if (this.right.dominoList[0].zero) {
                        end = 1;
                        skip = 0;
                }

                let parameter = [0];
                let tableau = this.left.clone();
                for (let number = size - 1; number >= end; --number) {
                        let rightDomino = this.right.dominoList[number];
                        parameter[number + skip] = tableau.removeRobinsonSchensted(rightDomino);
                }

                return new ParameterDominoRS({array: parameter});
        }
}

/**
 * This is a class of pairs of {TableauWithGrid}.
 * @extends {TableauPair}
 */
class TableauPairGrid extends TableauPair {
        /**
         * @param {Object} table
         * @param {string} table.type - "B", "C", or "D"
         * @param {Tableau} [table.left] - the left tableau of the pair
         * @param {Tableau} [table.right] - the right tableau of the pair
         */
        constructor(table) {
                super(table);
                if (!table.left) {
                        /**
                         * the left tableau of the pair
                         * @type {TableauWithGrid}
                         */
                        this.left = new TableauWithGrid({type: table.type});
                        /**
                         * the right tableau of the pair
                         * @type {TableauWithGrid}
                         */
                        this.right = new TableauWithGrid({type: table.type});
                } else if (!this.left.type) {
                        this.left = new TableauWithGrid({type, tableau: this.left});
                        this.right = new TableauWithGrid({type, tableau: this.right});
                }
        }

        /**
         * makes a deep copy
         * @return {TableauPairGrid}
         */
        clone() {
                let left = this.left.clone();
                let right = this.right.clone();
                return new TableauPairGrid({type: this.left.type, left, right});
        }

        /**
         * This static function performs the entire Domino Robinson-Schensted algorithm.
         * That is, it takes as input a signed permutation and outputs a pair of tableaux.
         * In this case, however, the output is a pair of TableauWithGrid.
         * @param {ParameterDominoRS} parameterObject - the signed permutation
         * wrapped in a ParameterDominoRS object.
         * @param {string} type - "B", "C", or "D"
         * @return {TableauPairGrid}
         */
        static RobinsonSchenstedGrid(parameterObject, type) {
                let parameter = parameterObject.parameter;
                let myTableauPair = new TableauPairGrid({type});
                for (let index = 1, length = parameter.length; index < length; ++index) {
                        myTableauPair.nextRobinsonSchensted(parameter[index], index);
                }

                return myTableauPair;
        }

        /**
         * This static function performs the Domino Robinson-Schensted algorithm,
         * just as the function RobinsonSchenstedGrid does.
         * So, again, the output is a pair of TableauWithGrid.
         * In addition, it draws each step.  So, the tableaux are drawn with their grid.
         * @param {ParameterDominoRS} parameterObject - the signed permutation
         * wrapped in a ParameterDominoRS object.
         * @return {TableauPairGrid}
         */
        static RobinsonSchenstedGridDrawSteps(parameterObject, type) {
                let parameter = parameterObject.parameter;
                for (let index = 1, length = parameter.length; index < length; ++index) {
                        Page.displayText(parameter[index]);
                        myTableauPair.nextRobinsonSchensted(parameter[index], index);
                        myTableauPair.draw();
                }

                return myTableauPair;
        }

        /**
         * This function moves the tableau with grid pair
         * through an open extended cycle specified by the back square
         * of a cycle in the left tableau.
         * @param {Square} backSquare - the back square of the cycle
         * @return {Object} {movedDominosLeft: {@link Array.<Array.<Domino>>},
         *  movedDominosRight: {@link Array.<Array.<Domino>>}}<br>
         * movedDominosLeft contains one array of Dominos for each cycle
         * in the left tableau which is part of the extended cycle.<br>
         * movedDominosRight contains one array of Dominos for each cycle
         * in the right tableau which is part of the extended cycle.
         */
        moveOpenExtendedCycleLeft(backSquare) {
                let currentBackSquare = backSquare.clone();
                let movedDominosLeft = [];
                let movedDominosRight = [];
                while (true) {
                        let moveDataLeft =
                                this.left.moveOpenCycle(currentBackSquare);
                        movedDominosLeft.push(moveDataLeft.movedDominos);
                        let moveDataRight =
                                this.right.moveOpenCycleToSquare(moveDataLeft.forwardSquare);
                        movedDominosRight.push(moveDataRight.movedDominos);
                        let newBackSquare = moveDataRight.backSquare;
                        if (newBackSquare.equals(backSquare)) {
                                break;
                        }

                        currentBackSquare = newBackSquare;
                }

                return { movedDominosLeft, movedDominosRight };
        }

        /**
         * This function moves the tableau with grid pair
         * through an open extended cycle specified by the forward square
         * of a cycle in the left tableau.
         * @param {Square} forwardSquare - the forward square of the cycle
         * @return {Object} {movedDominosLeft: {@link Array.<Array.<Domino>>},
         *  movedDominosRight: {@link Array.<Array.<Domino>>}}<br>
         * movedDominosLeft contains one array of Dominos for each cycle
         * in the left tableau which is part of the extended cycle.<br>
         * movedDominosRight contains one array of Dominos for each cycle
         * in the right tableau which is part of the extended cycle.
         */
        moveOpenExtendedCycleToSquareLeft(forwardSquare) {
                let currentForwardSquare = forwardSquare.clone();
                let movedDominosLeft = [];
                let movedDominosRight = [];
                while (true) {
                        let moveDataLeft =
                                this.left.moveOpenCycleToSquare(currentForwardSquare);
                        movedDominosLeft.push(moveDataLeft.movedDominos);
                        let moveDataRight =
                                this.right.moveOpenCycle(moveDataLeft.backSquare);
                        movedDominosRight.push(moveDataRight.movedDominos);
                        let newForwardSquare = moveDataRight.forwardSquare;
                        if (newForwardSquare.equals(forwardSquare)) {
                                break;
                        }

                        currentForwardSquare = newForwardSquare;
                }

                return { movedDominosLeft, movedDominosRight };
        }

        /**
         * This function moves the tableau with grid pair
         * through an open extended cycle specified by the back square
         * of a cycle in the right tableau.
         * @param {Square} backSquare - the back square of the cycle
         * @return {Object} {movedDominosLeft: {@link Array.<Array.<Domino>>},
         *  movedDominosRight: {@link Array.<Array.<Domino>>}}<br>
         * movedDominosLeft contains one array of Dominos for each cycle
         * in the left tableau which is part of the extended cycle.<br>
         * movedDominosRight contains one array of Dominos for each cycle
         * in the right tableau which is part of the extended cycle.
         */
        moveOpenExtendedCycleRight(backSquare) {
                let currentBackSquare = backSquare.clone();
                let movedDominosLeft = [];
                let movedDominosRight = [];
                while (true) {
                        let moveDataRight =
                                this.right.moveOpenCycle(currentBackSquare);
                        movedDominosRight.push(moveDataRight.movedDominos);
                        let moveDataLeft =
                                this.left.moveOpenCycleToSquare(moveDataRight.forwardSquare);
                        movedDominosLeft.push(moveDataLeft.movedDominos);
                        let newBackSquare = moveDataLeft.backSquare;
                        if (newBackSquare.equals(backSquare)) {
                                break;
                        }

                        currentBackSquare = newBackSquare;
                }

                return { movedDominosLeft, movedDominosRight };
        }

        /**
         * This function moves the tableau with grid pair
         * through an open extended cycle specified by the forward square
         * of a cycle in the right tableau.
         * @param {Square} forwardSquare - the forward square of the cycle
         * @return {Object} {movedDominosLeft: {@link Array.<Array.<Domino>>},
         *  movedDominosRight: {@link Array.<Array.<Domino>>}}<br>
         * movedDominosLeft contains one array of Dominos for each cycle
         * in the left tableau which is part of the extended cycle.<br>
         * movedDominosRight contains one array of Dominos for each cycle
         * in the right tableau which is part of the extended cycle.
         */
        moveOpenExtendedCycleToSquareRight(forwardSquare) {
                let currentForwardSquare = forwardSquare.clone();
                let movedDominosLeft = [];
                let movedDominosRight = [];
                while (true) {
                        let moveDataRight =
                                this.right.moveOpenCycleToSquare(currentForwardSquare);
                        movedDominosRight.push(moveDataRight.movedDominos);
                        let moveDataLeft =
                                this.left.moveOpenCycle(moveDataRight.backSquare);
                        movedDominosLeft.push(moveDataLeft.movedDominos);
                        let newForwardSquare = moveDataLeft.forwardSquare;
                        if (newForwardSquare.equals(forwardSquare)) {
                                break;
                        }

                        currentForwardSquare = newForwardSquare;
                }

                return { movedDominosLeft, movedDominosRight };
        }
}

/**
 * Class for drawing a single {@link Tableau} on a webpage.
 * The tableau to draw may be a {@link TableauWithGrid}, in which case it will be
 * drawn with its grid.  If not, it will be drawn with a fainter,
 * background grid of type C.
 */
class TableauRendererDOM {
        /**
         * @param {Object} table
         * @param {Tableau} table.tableau - the tableau to draw<br>
         */
        constructor(table) {
                /**
                 * the tableau to draw
                 * @type {TableauWithGrid|Tableau}
                 */
                this.tableau = table.tableau;
                /**
                 * the size of the side of the square
                 * alloted to each square of a Domino, in pixels
                 * @type {Number}
                 */
                this.gridSize = 30;
                if (this.tableau.type) {
                        /**
                         * a boolean which is true if tableau is a TableauWithGrid
                         * @type {Boolean}
                         */
                        this.hasGrid = true;
                        /**
                         * the x and y coordinates of the top-left corner
                         * of the tableau in relation to the 2x2 grid
                         * @type {Object}
                         */
                        this.offset = TableauWithGrid.getOffset(this.tableau.type);
                } else {
                        this.offset = {x: 0, y: 0};
                }
        }

        /**
         * This is helper function which makes the DOM element for
         * one domino in the tableau.
         * @param {Domino} domino - the domino to render
         * @return {Object} {dominoElement: Object, dominoText Object}<br>
         * dominoElement is the rendered domino<br>
         * dominoText is the text element contained in the
         * dominoElement.  Access to it is sometimes required
         * by animations.
         */
        makeDominoElement(domino) {
                let dominoElement = document.createElement('div');
                if (!domino.noBorder) {
                        dominoElement.className = "domino";
                } else {
                        dominoElement.className = "dominoNoBorder";
                }

                if (domino.zero) {
                        dominoElement.className += " dominoZero";
                } else if (domino.box) {
                        dominoElement.className += " dominoBox";
                } else if (domino.horizontal) {
                        dominoElement.className += " dominoHorizontal";
                } else {
                        dominoElement.className += " dominoVertical";
                }

                if (domino.highlight) {
                        dominoElement.className += " dominoHighlighted" + domino.highlight.toString();
                }

                let x = domino.x + this.offset.x;
                let y = domino.y + this.offset.y;

                dominoElement.style.left = x * this.gridSize + "px";
                dominoElement.style.top = y * this.gridSize + "px";

                let dominoText = document.createElement('div');
                dominoText.className = "dominoText";
                if (domino.noBorder) {
                        dominoText.className += " dominoTextNoBorder"
                }

                dominoText.innerHTML = domino.n;
                dominoElement.appendChild(dominoText);

                return { dominoElement, dominoText };
        }

        /**
         * This is a static version of makeDominoElement,
         * for use by renderers of pairs of tableaux.
         * @param {Domino} domino - the domino to render
         * @param {Object} offset -the x and y coordinates of the top-left corner
         * of the tableau in relation to the 2x2 grid
         * @param {number} gridSize -the size of the side of the square
         * alloted to each square of a Domino, in pixels
         * @return {Object} {dominoElement: Object, dominoText Object}<br>
         * dominoElement is the rendered domino<br>
         * dominoText is the text element contained in the
         * dominoElement.  Access to it is sometimes required
         * by animations.
         */
        static makeDominoElement(domino, offset, gridSize) {
                let dominoElement = document.createElement('div');
                if (!domino.noBorder) {
                        dominoElement.className = "domino";
                } else {
                        dominoElement.className = "dominoNoBorder";
                }

                if (domino.zero) {
                        dominoElement.className += " dominoZero";
                } else if (domino.box) {
                        dominoElement.className += " dominoBox";
                } else if (domino.horizontal) {
                        dominoElement.className += " dominoHorizontal";
                } else {
                        dominoElement.className += " dominoVertical";
                }

                if (domino.highlight) {
                        dominoElement.className += " dominoHighlighted" + domino.highlight.toString();
                }

                let x = domino.x + offset.x;
                let y = domino.y + offset.y;

                dominoElement.style.left = x * gridSize + "px";
                dominoElement.style.top = y * gridSize + "px";

                let dominoText = document.createElement('div');
                dominoText.className = "dominoText";
                if (domino.noBorder) {
                        dominoText.className += " dominoTextNoBorder"
                }

                dominoText.innerHTML = domino.n;
                dominoElement.appendChild(dominoText);

                return { dominoElement, dominoText};
        }

        /**
         * This is helper function which fills the DOM element for
         * the background grid with its 2x2 squares.
         * @param {Object} gridElement - the DOM element to fill with grid cells
         * @param {number} width - the width of the grid, an even integer
         * @param {number} height - the height of the grid, an even integer
         */
        fillGridElement(gridElement, width, height) {
                if (this.hasGrid) {
                        gridElement.className = "tableauGrid";
                } else {
                        gridElement.className = "tableauGridBackground";
                }

                gridElement.style.width = width * this.gridSize + "px";
                gridElement.style.height = height * this.gridSize + "px";

                for(let i = 0; i < height; i += 2) {
                        for(let j = 0; j < width; j += 2) {
                                let gridCell = document.createElement('div');
                                if (this.hasGrid) {
                                        gridCell.className = "tableauGridCell";
                                } else {
                                        gridCell.className = "tableauGridCellBackground";
                                }

                                gridCell.style.left = j * this.gridSize + "px";
                                gridCell.style.top  = i * this.gridSize + "px";

                                gridElement.appendChild(gridCell);
                        }
                }
        }

        /**
         * This is a static version of {@link TableauRendererDOM#fillGridElement},
         * for use by renderers of pairs of tableaux.
         * @param {Object} gridElement - the DOM element to fill with grid cells
         * @param {number} width - the width of the grid, an even integer
         * @param {number} height - the height of the grid, an even integer
         * @param {number} gridSize -the size of the side of the square
         * alloted to each square of a Domino, in pixels, thus, half the
         * size of the side of a 2x2 cell of the grid being rendered here.
         * @param {boolean} hasGrid - if false, the grid is rendered fainter.
         */
        static fillGridElement(gridElement, width, height, gridSize, hasGrid) {
                if (hasGrid) {
                        gridElement.className = "tableauGrid";
                } else {
                        gridElement.className = "tableauGridBackground";
                }

                gridElement.style.width = width * gridSize  + "px";
                gridElement.style.height = height * gridSize  + "px";

                for(let i = 0; i < height; i += 2) {
                        for(let j = 0; j < width; j += 2) {
                                let gridCell = document.createElement('div');
                                if (hasGrid) {
                                        gridCell.className = "tableauGridCell";
                                } else {
                                        gridCell.className = "tableauGridCellBackground";
                                }

                                gridCell.style.left = j * gridSize + "px";
                                gridCell.style.top  = i * gridSize + "px";

                                gridElement.appendChild(gridCell);
                        }
                }
        }

        /**
         * This function creates the DOM element for the {@link Tableau}.
         * @return {Object} the rendered tableau
         */
        renderDOM() {
                let wrapper = document.createElement('div');
                wrapper.className = "tableauRender";
                let gridElement = document.createElement('div');
                wrapper.appendChild(gridElement);

                for (let domino of this.tableau.dominoList) {
                        let dominoElement = this.makeDominoElement(domino).dominoElement;
                        wrapper.appendChild(dominoElement);
                }

                let width = this.tableau.getRowLength(0) + this.offset.x;
                let height = this.tableau.getColumnLength(0) +  this.offset.y;
                width = width % 2 == 0? width: ++width;
                height = height % 2 == 0? height: ++height;
                wrapper.style.width = width * this.gridSize + "px";
                wrapper.style.height = height * this.gridSize + "px";

                this.fillGridElement(gridElement, width, height);

                return wrapper;
        }
}

/**
 * Class for drawing a {@link TableauPair} on a webpage<br>
 * The {@link TableauPair} may be a {@link TableauPairGrid}.
 */
class TableauPairRendererDOM {
        /**
         * @param {Object} table
         * @param {TableauPair} table.tableauPair - the pair to draw
         */
        constructor(table) {
                /**
                 * the tableau pair to render
                 * @type {TableauPair}
                 */
                this.tableauPair = table.tableauPair;
        }

        /**
         * This function creates a DOM element to render a {@link TableauPair} of
         * tableaux side-by side.  It uses the {@link TableauRendererDOM} class for each
         * tableau in the pair, and wraps the results in a DOM element.
         * @return {Object} the rendered tableau pair
         */
        renderDOM() {
                let leftRenderer = new TableauRendererDOM({tableau: this.tableauPair.left});
                let rightRenderer = new TableauRendererDOM({tableau: this.tableauPair.right});
                let wrapperLeft = leftRenderer.renderDOM();
                let wrapperRight = rightRenderer.renderDOM();

                wrapperLeft.className += " tableauRenderLeft";
                wrapperRight.className += " tableauRenderRight";

                let wrapper = document.createElement('div');
                wrapper.className = "tableauPairRender";

                wrapper.appendChild(wrapperLeft);
                wrapper.appendChild(wrapperRight);

                return wrapper;
        }
}

function calc(strw) {
        let data = strw.split(" ").filter(x => x);
        let entry = [0];
        data.forEach((datum) => {
                entry.push(parseInt(datum));
        });
        let myTableau = new Tableau();
        for (let index = 1; index < entry.length; ++index) {
                myTableau.nextRobinsonSchensted(entry[index]);
        }
        return myTableau;
}
